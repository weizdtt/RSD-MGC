import os
import torch
from utils import mat_loader
from utils.my_dataset import My_dataset
import numpy as np
from torch.utils.data import DataLoader
from model.cycle_model import CycleModel
from utils.validator import Validator


class T_trainer:
    def __init__(self, opt):
        self.opt = opt
        opt.device = torch.device(opt.device)
        self.same_seeds(opt.seed)
        self.__get_loader(opt)
        opt.in_dims = self.in_dims
        opt.views = self.views
        opt.class_num = self.n_clusters
        self.model = CycleModel(opt)
        self.device = opt.device
        self.epochs = opt.epochs
        self.validator = Validator(self.views,
                                   self.data_size,
                                   self.dataset,
                                   self.n_clusters,
                                   opt.batch_size,
                                   self.device,
                                   self.model,
                                   opt)

    @staticmethod
    def same_seeds(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def __get_loader(self, opt):
        mat_data = getattr(mat_loader, opt.data_name) \
            (os.path.join('./data', opt.data_name)).get_data()
        self.dataset = My_dataset(mat_data)
        self.data_size = len(self.dataset)
        self.views = self.dataset.views
        self.in_dims = self.dataset.dims
        if opt.n_clusters == 0:
            self.n_clusters = self.dataset.n_clusters
        print("The number of classes in Data {} is: {}".format(opt.data_name, self.n_clusters))
        print("The total number of samples is {}".format(self.data_size))

        self.data_loader = DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=True)

    def train(self, epoch):
        self.model.train()
        total_losses = []
        for inputs, _ in self.data_loader:
            for v in range(self.views):
                inputs[v] = inputs[v].to(self.device)
            loss = self.model.optimizer(inputs)
            total_losses.append(loss)
        print('Training----->Epoch {}, Loss:{:.6f}'.format(epoch, sum(total_losses) / len(self.data_loader)))

    def running(self):
        epoch = 0
        best_result = {}
        print('+++++++++++++++++++++++++++++++++++current_lr: {}+++++++++++++++++++++++++++++++'.format(
            self.opt.learning_rate))
        print('+++++++++++++++++++++++++++++++++++current_seed: {}+++++++++++++++++++++++++++++++'.format(
            self.opt.seed))
        while epoch < self.epochs:
            self.train(epoch)
            best_result = self.validator.eval(epoch + 1)
            epoch += 1

        print('------------------Final Accuracy: {}------------------'.format(best_result['acc']))
        print('---------------------Final MNI: {}--------------------'.format(best_result['nmi']))
        print('---------------------Final ARI: {}--------------------'.format(best_result['ari']))
        print('-------------------Final Purity: {}-------------------'.format(best_result['pur']))
        return best_result

