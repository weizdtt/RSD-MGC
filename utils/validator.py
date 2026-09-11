import torch
import numpy as np
from sklearn.cluster import KMeans
from utils.metrics import evaluate
from torch.utils.data import DataLoader
from utils.helper import Preserving


class Validator:
    def __init__(self,
                 views,
                 data_size,
                 dataset,
                 n_clusters,
                 batch_size,
                 device,
                 model,
                 opt):
        self.views = views
        self.data_size = data_size
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.device = device
        self.model = model
        self.final_score_views = []
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.preserving = Preserving(self.model, opt)

    def __inference(self):
        self.model.eval()
        all_labels = []
        all_features = []
        eval_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for inputs, labels, in eval_loader:
            for v in range(len(inputs)):
                inputs[v] = inputs[v].to(self.device)
            with torch.no_grad():
                features = self.model.evaluation(inputs)
            all_features.append(features)
            all_labels.extend(labels.numpy())
        final_features = torch.cat(all_features, dim=0)
        return final_features, all_labels

    def eval(self, epoch):
        final_features, all_labels = self.__inference()
        labels = np.array(all_labels).reshape(self.data_size)
        pred_labels = self.kmeans.fit_predict(final_features.cpu().detach())
        acc, nmi, ari, pur = evaluate(labels, pred_labels)
        self.preserving([acc, nmi, ari, pur], [final_features, labels, pred_labels], epoch)
        return self.preserving.best_result
