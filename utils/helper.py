import os
import time
import torch


class Preserving:
    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
        self.best_result = {'acc': 0, 'nmi': 0, 'ari': 0, 'pur': 0}
        self.best_model = self.model.state_dict()
        self.epoch = 0
        self.preserving_time = time.strftime("%y_%m_%d_[%H_%M]", time.localtime())
        self.model_path = self.__create_dir()

    def __model_preserving(self):
        if self.opt.save_model:
            self.best_model = self.model.state_dict()
            print("------------------Model Saving------------------")
            torch.save(self.best_model, os.path.join(self.model_path, self.opt.data_name + '.pth'))

    def __result_comparing(self, result, features):
        if self.best_result['acc'] == 0:
            for ind, key in enumerate(self.best_result.keys()):
                self.best_result[key] = result[ind]
            self.__model_preserving()
        else:
            if self.best_result['acc'] < result[0]:
                for ind, key in enumerate(self.best_result.keys()):
                    self.best_result[key] = result[ind]
                self.__model_preserving()
            elif self.best_result['acc'] == result[0] and (self.best_result['nmi'] < result[1] or self.best_result['ari'] < result[2] or self.best_result['pur'] < result[3]):
                for ind, key in enumerate(self.best_result.keys()):
                    self.best_result[key] = result[ind]
                self.__model_preserving()

    def __create_dir(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')

        if self.opt.save_model:
            model_path = os.path.join(self.opt.save_path, 'model_' + self.preserving_time)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        else:
            model_path = None
        return model_path

    def __call__(self, result, features, epoch):
        self.epoch = epoch
        self.__result_comparing(result, features)
