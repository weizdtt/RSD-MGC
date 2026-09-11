from torch.utils.data import Dataset


class My_dataset(Dataset):
    def __init__(self, data):
        super(My_dataset, self).__init__()
        self.data = data
        self.views = len(self.data) - 1
        self.dims = [xs.shape[1] for xs in data[:self.views]]
        self.n_clusters = len(set(self.data[-1].numpy()))

    def __getitem__(self, index):
        sample = []
        for v in range(self.views):
            sample.append(self.data[v][index])
        return sample, self.data[-1][index]

    def __len__(self):
        return self.data[0].shape[0]
