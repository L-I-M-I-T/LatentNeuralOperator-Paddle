import json
import jsmin
import os
import torch
import numpy as np
import math
from matplotlib import pyplot as plt
from module.utils import *
from module.model import *


class Dict(dict):
    """
    Dictionary that allows to access per attributes and to except names from being loaded
    """
    def __init__(self, dictionary: dict = None):
        super(Dict, self).__init__()

        if dictionary is not None:
            self.load(dictionary)

    def __getattr__(self, item):
        try:
            return self[item] if item in self else getattr(super(Dict, self), item)
        except AttributeError:
            raise AttributeError(f'This dictionary has no attribute "{item}"')

    def load(self, dictionary: dict, name_list: list = None):
        """
        Loads a dictionary
        :param dictionary: Dictionary to be loaded
        :param name_list: List of names to be updated
        """
        for name in dictionary:
            data = dictionary[name]
            if name_list is None or name in name_list:
                if isinstance(data, dict):
                    if name in self:
                        self[name].load(data)
                    else:
                        self[name] = Dict(data)
                elif isinstance(data, list):
                    self[name] = list()
                    for item in data:
                        if isinstance(item, dict):
                            self[name].append(Dict(item))
                        else:
                            self[name].append(item)
                else:
                    self[name] = data

    def save(self, path):
        """
        Saves the dictionary into a json file
        :param path: Path of the json file
        """
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, 'cfg.json')

        with open(path, 'w') as file:
            json.dump(self, file, indent=True)


class Configuration(Dict):
    """
    Configuration loaded from a json file
    """
    def __init__(self, path: str, default_path=None):
        super(Configuration, self).__init__()

        if default_path is not None:
            self.load(default_path)

        self.load(path)

    def load_model(self, path: str):
        self.load(path, name_list=["model"])

    def load(self, path: str, name_list: list = None):
        """
        Loads attributes from a json file
        :param path: Path of the json file
        :param name_list: List of names to be updated
        :return:
        """
        with open(path) as file:
            data = json.loads(jsmin.jsmin(file.read()))

            super(Configuration, self).load(data, name_list)


class Checkpoint():
    def __init__(self, dir, model, device):
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            
        self.model = model
        self.device = device
    
    def load(self, epoch):
        self.model.load_state_dict(torch.load("{}/{}.pt".format(self.dir, epoch)), map_location="cuda:{}".format(self.device))
    
    def save(self, epoch):
        torch.save(self.model.state_dict(),  "{}/{}.pt".format(self.dir, epoch))


def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    torch.backends.cudnn.deterministic = True


def get_num_params(model):
    total_num = 0
    for p in list(model.parameters()):
        num = p.size() + (2,) if p.is_complex() else p.size()
        total_num = total_num + math.prod(num)
    return total_num


class Null():
    def __init__(self, attr=None):
        self.attr = None
    
    def step(self):
        return


def logger(log_dir, np_list, str_list):
    for i in range(0, len(np_list)):
        np_list[i] = np.array(np_list[i])
        np.save(log_dir + "/" + str_list[i], np_list[i])
    
    for i in range(1, len(np_list)):
        plt.xlabel(str_list[0])
        plt.ylabel(str_list[i])
        plt.plot(np_list[0], np.log10(np_list[i]), label=str_list[i])
        plt.legend()
        plt.savefig(log_dir + "/" + str_list[i] + ".png")
        plt.clf()


def draw_1D(x, y, hold=True, x_label=None, y_label=None, filename=None):
    plt.plot(x, y["value"], label=y["label"])
    if not hold:
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.savefig(filename)
        plt.clf()


def draw_2D(x, y, u, x_label, y_label, u_label, filename=None):
    c = plt.pcolormesh(x, y, u, cmap='rainbow', shading='gouraud')
    plt.colorbar(c, label=u_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename, dpi=300)
    plt.clf()
    

def show_Burgers(u, filename):
    u = torch.squeeze(u, -1)
    u = u.cpu().numpy()
    x = np.array([np.linspace(0, u.shape[0], u.shape[0])])
    x = np.repeat(x, [u.shape[1]], axis=0)
    y = np.array([np.linspace(0, u.shape[1], u.shape[1])])
    y = np.repeat(y, [u.shape[0]], axis=0).T
    draw_2D(x, y, u, "x", "t", "u", filename)


def padding(x, max_len):
    device = x.device
    pad = torch.zeros((x.shape[0], max_len - x.shape[1], x.shape[2])).to(device)
    x = torch.cat((x, pad), dim=1)
    return x


def data_preprocess_completer_DeepONet(masker, x, y, device):
    mask, vertex0, vertex1 = masker.get()
    
    x = x.numpy()
    y = y.numpy()
    
    mask_idx = list(mask.to_sparse().indices().transpose(0,1).numpy())
    
    x_ob = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
    y_ob = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
    ob = torch.cat((x_ob, y_ob), dim=-1)
    
    x = torch.tensor(x[(slice(None), *tuple(slice(r0, r1) for r0, r1 in zip(vertex0, vertex1)), slice(None))]).to(device)
    y = torch.tensor(y[(slice(None), *tuple(slice(r0, r1) for r0, r1 in zip(vertex0, vertex1)), slice(None))]).to(device)
    x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
    y = torch.reshape(y, (y.shape[0], -1, y.shape[-1]))
    return x, y, ob


def data_preprocess_completer_GNOT(masker, x, y, device):
    mask, vertex0, vertex1 = masker.get()
    x = x.numpy()
    y = y.numpy()
    
    mask_idx = list(mask.to_sparse().indices().transpose(0,1).numpy())
    
    x_ob = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
    y_ob = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
    ob = torch.cat((x_ob, y_ob), dim=-1)
    
    x = torch.tensor(x[(slice(None), *tuple(slice(r0, r1) for r0, r1 in zip(vertex0, vertex1)), slice(None))]).to(device)
    y = torch.tensor(y[(slice(None), *tuple(slice(r0, r1) for r0, r1 in zip(vertex0, vertex1)), slice(None))]).to(device)
    x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
    y = torch.reshape(y, (y.shape[0], -1, y.shape[-1]))
    return x, y, ob


def data_preprocess_completer_LNO(masker, x, y, device):
    mask, vertex0, vertex1 = masker.get()
    x = x.numpy()
    y = y.numpy()
    
    mask_idx = list(mask.to_sparse().indices().transpose(0,1).numpy())
    
    x_ob = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
    y_ob = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
    ob = torch.cat((x_ob, y_ob), dim=-1)
    
    x = torch.tensor(x[(slice(None), *tuple(slice(r0, r1) for r0, r1 in zip(vertex0, vertex1)), slice(None))]).to(device)
    y = torch.tensor(y[(slice(None), *tuple(slice(r0, r1) for r0, r1 in zip(vertex0, vertex1)), slice(None))]).to(device)
    x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
    y = torch.reshape(y, (y.shape[0], -1, y.shape[-1]))
    return x, y, ob


def data_preprocess_propagator_DeepONet(masker, x, y, device):
    mask, _, _ = masker.reset().get()
                
    x_list = []
    y_list = []
    ob_list = []
    x = x.numpy()
    y = y.numpy()

    max_len = int(torch.sum(mask[-1]))
    
    for i in range(0, mask.shape[0]-1):
        mask_idx = list(mask[i].to_sparse().indices().transpose(0,1).numpy())
        
        x_ob = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        y_ob = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        ob = torch.cat((x_ob, y_ob), dim=-1)
        
        mask_idx = list(mask[i+1].to_sparse().indices().transpose(0,1).numpy())
        
        x_query = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        y_query = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        
        x_list.append(x_query)
        y_list.append(y_query)
        ob_list.append(ob)
    
    if not masker.need_initial_predict():
        x_list = x_list[1:]
        y_list = y_list[1:]
        ob_list = ob_list[1:]

    return x_list, y_list, ob_list


def data_preprocess_propagator_GNOT(masker, x, y, device):
    mask, _, _ = masker.reset().get()
                
    x_list = []
    y_list = []
    ob_list = []
    x = x.numpy()
    y = y.numpy()

    max_len = int(torch.sum(mask[-1]))
    
    for i in range(0, mask.shape[0]-1):
        mask_idx = list(mask[i].to_sparse().indices().transpose(0,1).numpy())
        
        x_ob = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        y_ob = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        ob = torch.cat((x_ob, y_ob), dim=-1)
        
        mask_idx = list(mask[i+1].to_sparse().indices().transpose(0,1).numpy())
        
        x_query = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        y_query = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        
        x_list.append(x_query)
        y_list.append(y_query)
        ob_list.append(ob)
    
    if not masker.need_initial_predict():
        x_list = x_list[1:]
        y_list = y_list[1:]
        ob_list = ob_list[1:]

    return x_list, y_list, ob_list


def data_preprocess_propagator_LNO(masker, x, y, device):
    mask, _, _ = masker.reset().get()
                
    x_list = []
    y_list = []
    ob_list = []
    x = x.numpy()
    y = y.numpy()

    max_len = int(torch.sum(mask[-1]))
    
    for i in range(0, mask.shape[0]-1):
        mask_idx = list(mask[i].to_sparse().indices().transpose(0,1).numpy())
        
        x_ob = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        y_ob = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        ob = torch.cat((x_ob, y_ob), dim=-1)
        
        mask_idx = list(mask[i+1].to_sparse().indices().transpose(0,1).numpy())
        
        x_query = torch.tensor(np.array([x[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        y_query = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in mask_idx])).float().transpose(0, 1).to(device)
        
        x_list.append(x_query)
        y_list.append(y_query)
        ob_list.append(ob)
    
    if not masker.need_initial_predict():
        x_list = x_list[1:]
        y_list = y_list[1:]
        ob_list = ob_list[1:]

    return x_list, y_list, ob_list


class Dataset(torch.utils.data.Dataset):
    class transformer_LN_like():
        def __init__(self, x, y, f):
            self.x_mean = torch.mean(x, dim=tuple(range(0, len(x.shape)-1)))
            self.x_std = torch.std(x, dim=tuple(range(0, len(x.shape)-1))) + 1e-8
            self.y_mean = torch.mean(y, dim=tuple(range(0, len(y.shape)-1)))
            self.y_std = torch.std(y, dim=tuple(range(0, len(y.shape)-1))) + 1e-8
            self.f_mean = torch.mean(f, dim=tuple(range(0, len(f.shape)-1)))
            self.f_std = torch.std(f, dim=tuple(range(0, len(f.shape)-1))) + 1e-8
        
        def apply(self, data, mean, std, inverse=False):
            mean = mean.to(data.device)
            std = std.to(data.device)
            shape = data.shape
            data = torch.reshape(data, (-1, *tuple(mean.shape)))
            if not inverse:
                data = (data - mean) / std
            else:
                data = data * std + mean
            data = torch.reshape(data, shape) 
            return data
        
        def apply_x(self, x, inverse=False):
            return self.apply(x, self.x_mean, self.x_std, inverse)
        
        def apply_y(self, y, inverse=False):
            return self.apply(y, self.y_mean, self.y_std, inverse)
        
        def apply_f(self, f, inverse=False):
            return self.apply(f, self.f_mean, self.f_std, inverse)
        
    class transformer_BN_like():
        def __init__(self, x, y, f):
            self.y_mean = torch.mean(y, dim=0)
            self.y_std = torch.std(y, dim=0) + 1e-8
            self.f_mean = torch.mean(f, dim=0)
            self.f_std = torch.std(f, dim=0) + 1e-8
            
        def apply(self, data, mean, std, inverse=False):
            mean = mean.to(data.device)
            std = std.to(data.device)
            shape = data.shape
            data = torch.reshape(data, (-1, *tuple(mean.shape)))
            if not inverse:
                data = (data - mean) / std
            else:
                data = data * std + mean
            data = torch.reshape(data, shape)
            return data
        
        def apply_x(self, x, inverse=False):
            if not inverse:
                x = x * 2 - 1
            else:
                x = (x + 1) / 2
            return x
        
        def apply_y(self, y, inverse=False):
            return self.apply(y, self.y_mean, self.y_std, inverse)
        
        def apply_f(self, f, inverse=False):
            return self.apply(f, self.f_mean, self.f_std, inverse)
        
    class transformer_none():
        def __init__(self):
            pass
        
        def apply_x(self, x, inverse=False):
            return x
        
        def apply_y(self, y, inverse=False):
            return y
        
        def apply_f(self, f, inverse=False):
            return f
    
    def __init__(self, data_name, mode, transformer):
        super().__init__()
        data_file = "../datas/" + data_name + "_" + mode + ".npy"
        self.dataset = np.load(data_file, allow_pickle=True).tolist()
        self.x = torch.tensor(self.dataset['x']).float()
        self.y = torch.tensor(self.dataset['y']).float()
        self.f = torch.tensor(self.dataset['f']).float()
        self.l = self.y.shape[0]
        
        if isinstance (transformer, str):
            if transformer == "LN":
                self.transformer = Dataset.transformer_LN_like(self.x, self.y, self.f)
            elif transformer == "BN":
                self.transformer = Dataset.transformer_BN_like(self.x, self.y, self.f)
            else:
                raise NotImplementedError("Invalid Transformer !")
        else:
            self.transformer = transformer
            
        self.x = self.transformer.apply_x(self.x)
        self.y = self.transformer.apply_y(self.y)
        self.f = self.transformer.apply_f(self.f)

    def __len__(self):
        return self.l

    def __getitem__(self, idx):   
        return self.x[idx], self.y[idx], self.f[idx]
    
    def shape(self):
        return self.x.shape[1:-1]
    
    def dim(self):
        return self.x.shape[-1], self.y.shape[-1], self.f.shape[-1]
    
    def get_transformer(self):
        return self.transformer


class Masker_Propagator_Random():
    def __init__(self, shape, initial_region, initial_ratio, initial_predict, series_length):
        self.shape = np.array(shape)
        self.mask = np.zeros((series_length+1, *shape))
        self.region = [initial_region + (np.ones_like(initial_region) - initial_region) * r for r in list(np.linspace(0, 1, series_length))]
        
        self.vertex0 = ((self.shape - self.shape * initial_region) / 2).astype(int)
        self.vertex1 = (self.shape - (self.shape - self.shape * initial_region) / 2).astype(int)
        
        self.region_mask = np.zeros(tuple(self.vertex1 - self.vertex0)).flatten()
        self.region_mask[0:int(math.prod(tuple(self.vertex1 - self.vertex0)) * initial_ratio)] = 1
        
        if initial_predict == "Y":
            self.initial_predict = True
        elif initial_predict == "N":
            self.initial_predict = False
        else:
            raise NotImplementedError("Invalid Initial Predict")
        
        self.reset()
        
        for i in range(1, series_length+1):
            vertex0 = ((self.shape - self.shape * self.region[i-1]) / 2).astype(int)
            vertex1 = (self.shape - (self.shape - self.shape * self.region[i-1]) / 2).astype(int)
            self.mask[i][tuple(slice(r0, r1) for r0, r1 in zip(list(vertex0), list(vertex1)))] = 1

    def reset(self):
        np.random.shuffle(self.region_mask)
        self.mask[0][tuple(slice(r0, r1) for r0, r1 in zip(list(self.vertex0), list(self.vertex1)))] \
        = np.reshape(self.region_mask, tuple(self.vertex1 - self.vertex0))
        return self
    
    def get(self):
        return torch.tensor(self.mask), self.vertex0, self.vertex1
    
    def need_initial_predict(self):
        return self.initial_predict
    
    
class Masker_Completer_Random():
    def __init__(self, shape, initial_region, initial_ratio):
        self.shape = np.array(shape)
        self.mask = np.zeros(shape)
        self.vertex0 = ((self.shape - self.shape * initial_region) / 2).astype(int)
        self.vertex1 = (self.shape - (self.shape - self.shape * initial_region) / 2).astype(int)
        self.region_mask = np.zeros(tuple(self.vertex1 - self.vertex0)).flatten()
        self.region_mask[0:int(math.prod(tuple(self.vertex1 - self.vertex0)) * initial_ratio)] = 1
        self.reset()

    def reset(self):
        np.random.shuffle(self.region_mask)
        self.mask[tuple(slice(r0, r1) for r0, r1 in zip(list(self.vertex0), list(self.vertex1)))] \
        = np.reshape(self.region_mask, tuple(self.vertex1 - self.vertex0))
        return self
    
    def get(self):
        self.reset()
        return torch.tensor(self.mask), self.vertex0, self.vertex1


class Masker_Completer_Fix():
    def __init__(self, shape, initial_region, sample_steps):
        self.shape = np.array(shape)
        self.mask = np.zeros(shape)
        self.vertex0 = ((self.shape - self.shape * initial_region) / 2).astype(int)
        self.vertex1 = (self.shape - (self.shape - self.shape * initial_region) / 2).astype(int)
        self.mask[tuple(slice(r0, r1, step) for r0, r1, step in zip(list(self.vertex0), list(self.vertex1), list(sample_steps)))] = 1
    
    def get(self):
        return torch.tensor(self.mask), self.vertex0, self.vertex1


class Poser_Propagator():
    def __init__(self, shape, region):
        if min(region) < 1.0:
            self.pos = torch.zeros(shape)
            for dim in range(len(region)):
                if  region[dim] < 1.0:
                    idx_head = [slice(None) for _ in range(0, len(shape))]
                    idx_tail = [slice(None) for _ in range(0, len(shape))]
                    idx_head[dim] = 0
                    idx_tail[dim] = -1
                    self.pos[idx_head] = 1
                    self.pos[idx_tail] = 1
        else:
            self.pos = torch.ones(shape)

    def get(self):
        return self.pos


class Poser_Completer():
    def __init__(self, shape, region):
        self.pos = torch.ones(list((np.array(shape) * np.array(region)).astype(int)))

    def get(self):
        return self.pos


class RelLpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(RelLpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):    
        error = torch.mean(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        target = torch.mean(abs(target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        rloss = torch.mean(error / target)
        return rloss


class LpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.mean(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        loss = torch.mean(error)
        return loss
    

class MpELoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(MpELoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.mean(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape))))
        loss = torch.mean(error)
        return loss
    
    
class RelMpELoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(RelMpELoss, self).__init__()
        self.p = p
        
    def forward(self, pred, target):
        error = torch.mean(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape))))
        target = torch.mean(abs(target) ** self.p, tuple(range(1, len(pred.shape))))
        rloss = torch.mean(error / target)
        return rloss


class Scheduler_Customized():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.epoch = 0
    
    def step(self):
        self.epoch = self.epoch + 1
        lr = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def get_data_model(config, device):
    train_dataset = Dataset(config.data.name, "train", config.data.transformer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler, 
        batch_size=config.data.train_batch_size, 
        drop_last=True, 
        pin_memory=True
        )

    transformer = train_dataset.get_transformer()

    val_dataset = Dataset(config.data.name, "val", transformer)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=val_dataset, 
        sampler=val_sampler, 
        batch_size=config.data.val_batch_size, 
        drop_last=True, 
        pin_memory=True
        )
    
    test_dataset = Dataset(config.data.name, "test", transformer)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=test_dataset, 
        sampler=test_sampler, 
        batch_size=1, 
        drop_last=True, 
        pin_memory=True
        )
    
    x_dim, y_dim, f_dim = train_dataset.dim()
    
    if config.observation.method == "random":
        if config.role == "propagator":
            masker = Masker_Propagator_Random(train_dataset.shape(), config.observation.initial_region, config.observation.initial_ratio, config.observation.initial_predict, config.observation.series_length)
        elif config.role == "completer":
            masker = Masker_Completer_Random(train_dataset.shape(), config.observation.initial_region, config.observation.initial_ratio)
        else:
            raise NotImplementedError("Invalid Role !")
    elif config.observation.method == "fix":
        if config.role == "completer":
            masker = Masker_Completer_Fix(train_dataset.shape(), config.observation.initial_region, config.observation.sample_steps)
    else:
        raise NotImplementedError("Invalid Observation Method !")
    
    if config.role == "propagator":
        poser = Poser_Propagator(train_dataset.shape(), config.observation.initial_region)
    elif config.role == "completer":
        poser = Poser_Completer(train_dataset.shape(), config.observation.initial_region)
    else:
        raise NotImplementedError("Invalid Role !")
    
    if config.model.name == "GNOT":
        model = GNOT(config.model.n_block, config.model.n_dim, config.model.n_head, config.model.n_layer, config.model.n_expert, 
                    x_dim, y_dim, f_dim, config.model.attn, config.model.act).to(device)
    elif config.model.name == "DeepONet":
        model = DeepONet(x_dim, y_dim, config.model.n_dim).to(device)
    elif config.model.name=="LNO":
        model = LNO(config.model.n_block, config.model.n_mode, config.model.n_dim, config.model.n_head, config.model.n_layer, 
                x_dim, f_dim, y_dim, config.model.attn, config.model.act).to(device)
    else:
        raise NotImplementedError("Invalid Model !")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    
    if config.loss.name == "L2":
        loss = LpLoss(p=2).to(device)
    elif config.loss.name == "L1":
        loss = LpLoss(p=1).to(device)
    elif config.loss.name == "rL2":
        loss = RelLpLoss(p=2).to(device)
    elif config.loss.name == "rL1":
        loss = RelLpLoss(p=1).to(device)
    elif config.loss.name == "MSE":
        loss = MpELoss(p=2).to(device)
    elif config.loss.name == "MAE":
        loss = MpELoss(p=1).to(device)
    else:
        raise NotImplementedError("Invalid Loss !")
    
    if config.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
    elif config.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=(config.optimizer.beta0, config.optimizer.beta1))
    elif config.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.optimizer.lr)
    else:
        raise NotImplementedError("Invalid Optimizer !")
    
    if config.scheduler.name == "Customized":
        scheduler = Scheduler_Customized(optimizer)
    elif config.scheduler.name == "Step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler.step_size*len(train_dataloader), gamma=config.scheduler.gamma)
    elif config.scheduler.name == "CosRestart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.T_0*len(train_dataloader), T_mult=config.scheduler.T_mult)
    elif config.scheduler.name == "Cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler.T_max*len(train_dataloader))
    elif config.scheduler.name == "OneCycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.optimizer.lr, div_factor=config.scheduler.div_factor, 
                                                        steps_per_epoch=len(train_dataloader), epochs=config.train.epoch)
    else:
        raise NotImplementedError("Invalid Scheduler !")
    
    return train_dataset, train_sampler, train_dataloader, \
           val_dataset, val_sampler, val_dataloader, \
           test_dataset, test_sampler, test_dataloader, \
           transformer, masker, poser, \
           model, loss, optimizer, scheduler
