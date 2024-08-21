import paddle
import numpy as np


class LNO_Dataset(paddle.io.Dataset):
    class normalizer():
        def __init__(self, x, y1, y2):
            self.x_flag = False
            self.y1_flag = False
            self.y2_flag = False
            old_x_shape = x.shape
            old_y1_shape = y1.shape
            old_y2_shape = y2.shape
            x = paddle.reshape(x, (-1, x.shape[-1]))
            y1 = paddle.reshape(y1, (-1, y1.shape[-1]))
            y2 = paddle.reshape(y2, (-1, y2.shape[-1]))
            self.x_mean = paddle.mean(x, axis=0)
            self.x_std = paddle.std(x, axis=0) + 1e-8
            self.y1_mean = paddle.mean(y1, axis=0)
            self.y1_std = paddle.std(y1, axis=0) + 1e-8
            self.y2_mean = paddle.mean(y2, axis=0)
            self.y2_std = paddle.std(y2, axis=0) + 1e-8
            x = paddle.reshape(x, old_x_shape)
            y1 = paddle.reshape(y1, old_y1_shape)
            y2 = paddle.reshape(y2, old_y2_shape)
        
        def is_apply_x(self):
            return self.x_flag
        
        def is_apply_y1(self):
            return self.y1_flag
        
        def is_apply_y2(self):
            return self.y2_flag
        
        def apply_x(self, x, device, inverse=False):
            self.x_mean = self.x_mean.to(device)
            self.x_std = self.x_std.to(device)
            
            old_x_shape = x.shape
            x = paddle.reshape(x, (-1, x.shape[-1]))
            if not inverse:
                x = (x - self.x_mean) / self.x_std
                self.x_flag = True
            else:
                x = x * self.x_std + self.x_mean
            x = paddle.reshape(x, old_x_shape)
            return x

        def apply_y1(self, y1, device, inverse=False):
            self.y1_mean = self.y1_mean.to(device)
            self.y1_std = self.y1_std.to(device)
            
            old_y1_shape = y1.shape
            y1 = paddle.reshape(y1, (-1, y1.shape[-1]))
            if not inverse:
                y1 = (y1 - self.y1_mean) / self.y1_std
                self.y1_flag = True
            else:
                y1 = y1 * self.y1_std + self.y1_mean
            y1 = paddle.reshape(y1, old_y1_shape)
            return y1
        
        def apply_y2(self, y2, device, inverse=False):
            self.y2_mean = self.y2_mean.to(device)
            self.y2_std = self.y2_std.to(device)
            
            old_y2_shape = y2.shape
            y2 = paddle.reshape(y2, (-1, y2.shape[-1]))
            if not inverse:
                y2 = (y2 - self.y2_mean) / self.y2_std
                self.y2_flag = True
            else:
                y2 = y2 * self.y2_std + self.y2_mean
            y2 = paddle.reshape(y2, old_y2_shape)
            return y2

    def __init__(self, data_name, data_mode, data_normalize, data_concat):
        super().__init__()
        data_file = "./datas/{}_{}.npy".format(data_name, data_mode)
        self.dataset = np.load(data_file, allow_pickle=True).tolist()
        self.x = paddle.to_tensor(self.dataset['x'], dtype="float32")
        self.y1 = paddle.to_tensor(self.dataset['y1'], dtype="float32")
        if data_concat:
            self.y1 = paddle.concat((self.x, self.y1), axis=-1)
        self.y2 = paddle.to_tensor(self.dataset['y2'], dtype="float32")
        self.normalizer = LNO_Dataset.normalizer(self.x, self.y1, self.y2)
        if data_normalize:
            self.x = self.normalizer.apply_x(self.x, "cpu")
            self.y1 = self.normalizer.apply_y1(self.y1, "cpu")
            self.y2 = self.normalizer.apply_y2(self.y2, "cpu")
        self.l = self.x.shape[0]

    def __len__(self):
        return self.l

    def __getitem__(self, idx):   
        return self.x[idx], self.y1[idx], self.y2[idx], 

