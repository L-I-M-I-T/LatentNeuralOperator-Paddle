import torch
import numpy as np


class LNO_dataset(torch.utils.data.Dataset):
    class Normalizer():
        def __init__(self, x, y1, y2):
            self.x_flag = False
            self.y1_flag = False
            self.y2_flag = False
            old_x_shape = x.shape
            old_y1_shape = y1.shape
            old_y2_shape = y2.shape
            x = torch.reshape(x, (-1, x.shape[-1]))
            y1 = torch.reshape(y1, (-1, y1.shape[-1]))
            y2 = torch.reshape(y2, (-1, y2.shape[-1]))
            self.x_mean = torch.mean(x, dim=0)
            self.x_std = torch.std(x, dim=0) + 1e-8
            self.y1_mean = torch.mean(y1, dim=0)
            self.y1_std = torch.std(y1, dim=0) + 1e-8
            self.y2_mean = torch.mean(y2, dim=0)
            self.y2_std = torch.std(y2, dim=0) + 1e-8
            x = torch.reshape(x, old_x_shape)
            y1 = torch.reshape(y1, old_y1_shape)
            y2 = torch.reshape(y2, old_y2_shape)
        
        def is_apply_x(self):
            return self.x_flag
        
        def is_apply_y1(self):
            return self.y1_flag
        
        def is_apply_y2(self):
            return self.y2_flag
        
        def apply_x(self, x, inverse=False):
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
            
            old_x_shape = x.shape
            x = torch.reshape(x, (-1, x.shape[-1]))
            if not inverse:
                x = (x - self.x_mean) / self.x_std
                self.x_flag = True
            else:
                x = x * self.x_std + self.x_mean
            x = torch.reshape(x, old_x_shape)
            return x

        def apply_y1(self, y1, inverse=False):
            self.y1_mean = self.y1_mean.to(y1.device)
            self.y1_std = self.y1_std.to(y1.device)
            
            old_y1_shape = y1.shape
            y1 = torch.reshape(y1, (-1, y1.shape[-1]))
            if not inverse:
                y1 = (y1 - self.y1_mean) / self.y1_std
                self.y1_flag = True
            else:
                y1 = y1 * self.y1_std + self.y1_mean
            y1 = torch.reshape(y1, old_y1_shape)
            return y1
        
        def apply_y2(self, y2, inverse=False):
            self.y2_mean = self.y2_mean.to(y2.device)
            self.y2_std = self.y2_std.to(y2.device)
            
            old_y2_shape = y2.shape
            y2 = torch.reshape(y2, (-1, y2.shape[-1]))
            if not inverse:
                y2 = (y2 - self.y2_mean) / self.y2_std
                self.y2_flag = True
            else:
                y2 = y2 * self.y2_std + self.y2_mean
            y2 = torch.reshape(y2, old_y2_shape)
            return y2

    def __init__(self, data_name, data_mode):
        super().__init__()
        data_file = "./datas/{}_{}.npy".format(data_name, data_mode)
        self.dataset = np.load(data_file, allow_pickle=True).tolist()
        self.x = torch.tensor(self.dataset['x']).float()
        self.y1 = torch.tensor(self.dataset['y1']).float()
        if data_name in ["Darcy", "Plasticity", "Airfoil", "Pipe", "NS2d"]:
            self.y1 = torch.cat((self.x, self.y1), dim=-1)
        self.y2 = torch.tensor(self.dataset['y2']).float()
        self.normalizer = LNO_dataset.Normalizer(self.x, self.y1, self.y2)
        if data_name in ["Darcy", "Elasticity"]:
            self.x = self.normalizer.apply_x(self.x)
            self.y1 = self.normalizer.apply_y1(self.y1)
            self.y2 = self.normalizer.apply_y2(self.y2)
        self.l = self.x.shape[0]

    def __len__(self):
        return self.l

    def __getitem__(self, idx):   
        return self.x[idx], self.y1[idx], self.y2[idx], 
    
    def dim(self):
        return self.x.shape[-1], self.y1.shape[-1], self.y2.shape[-1]

    def get_normalizer(self):
        return self.normalizer
