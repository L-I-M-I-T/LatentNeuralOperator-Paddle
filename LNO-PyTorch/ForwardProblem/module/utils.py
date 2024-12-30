import json
import jsmin
import os
import torch
import numpy as np
import math
from module.dataset import *
from module.model import *
from module.loss import *
from matplotlib import pyplot as plt


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


class Logger():
    def __init__(self, dir):
        self.dir = dir
        self.f = open(self.dir + "log.txt", "w")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            
    def print(self, st):
        print(st, file=self.f, flush=True)
    
    def save(self, np_list, str_list):
        for i in range(0, len(np_list)):
            np_list[i] = np.array(np_list[i])
            np.save(self.dir + "/" + str_list[i], np_list[i])
        
        for i in range(1, len(np_list)):
            plt.xlabel(str_list[0])
            plt.ylabel(str_list[i])
            plt.plot(np_list[0], np.log10(np_list[i]), label=str_list[i])
            plt.legend()
            plt.savefig(self.dir + "/" + str_list[i] + ".png")
            plt.clf()


class Scheduler_NULL():
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def step(self):
        return


def save_para(arg, config):
    para_dir = "./experiments/{}/para/".format(arg.exp)
    if not os.path.exists(para_dir):
        os.makedirs(para_dir)
    json.dump(arg.__dict__, open(para_dir + "arg.json", "w"), indent=2)
    json.dump(config, open(para_dir +"config.json", "w"), indent=2)


def get_model_data(config, model_attr, device):
    train_dataset = LNO_dataset(config.data.name, "train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler, 
        batch_size=config.data.train_batch_size, 
        drop_last=True, 
        pin_memory=True,
        shuffle=False
        )

    normalizer = train_dataset.get_normalizer()
    
    val_dataset = LNO_dataset(config.data.name, "val")
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=val_dataset, 
        sampler=val_sampler, 
        batch_size=config.data.val_batch_size, 
        drop_last=True, 
        pin_memory=True,
        shuffle=False
        )
    
    test_dataset = val_dataloader
    test_sampler = val_sampler
    test_dataloader = val_dataloader
    
    x_dim, y1_dim, y2_dim = train_dataset.dim()
    if config.model.name == "LNO":
        model = LNO(config.model.n_block, config.model.n_mode, config.model.n_dim, config.model.n_head, config.model.n_layer, 
                    x_dim, y1_dim, y2_dim, config.model.attn, config.model.act, model_attr).to(device)
    elif config.model.name == "LNO_single":
        model = LNO_single(config.model.n_block, config.model.n_mode, config.model.n_dim, config.model.n_head, config.model.n_layer, 
                    x_dim, y1_dim, y2_dim, config.model.attn, config.model.act, model_attr).to(device)
    elif config.model.name == "LNO_triple":
        model = LNO_triple(config.model.n_block, config.model.n_mode, config.model.n_dim, config.model.n_head, config.model.n_layer, 
                    x_dim, y1_dim, y2_dim, config.model.attn, config.model.act, model_attr).to(device)
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
    else:
        raise NotImplementedError("Invalid Loss !")
    
    if config.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=(config.optimizer.beta0, config.optimizer.beta1))
    elif config.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=(config.optimizer.beta0, config.optimizer.beta1))
    elif config.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.optimizer.lr)
    else:
        raise NotImplementedError("Invalid Optimizer !")
    
    if config.scheduler.name == "NULL":
        scheduler = Scheduler_NULL(optimizer)
    elif config.scheduler.name == "Step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler.step_size*len(train_dataloader), gamma=config.scheduler.gamma)
    elif config.scheduler.name == "CosRestart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.T_0*len(train_dataloader), T_mult=config.scheduler.T_mult)
    elif config.scheduler.name == "Cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler.T_max*len(train_dataloader))
    elif config.scheduler.name == "OneCycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.optimizer.lr, pct_start=config.scheduler.pct_start, 
                                                        div_factor=config.scheduler.div_factor, final_div_factor=config.scheduler.final_div_factor,
                                                        steps_per_epoch=len(train_dataloader), epochs=config.train.epoch)
    else:
        raise NotImplementedError("Invalid Scheduler !")
    
    return train_dataloader, val_dataloader, test_dataloader, \
           normalizer, model, loss, optimizer, scheduler
