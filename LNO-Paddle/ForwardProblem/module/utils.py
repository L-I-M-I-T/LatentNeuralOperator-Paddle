import paddle
import numpy as np
import random
import math
import json
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def save_para(arg, filename):
    if not os.path.exists("./experiments/"):
        os.mkdir("./experiments/")
    fp = open(filename, "w")
    dict = vars(arg)
    json.dump(dict, fp)
    fp.close()


def get_num_params(model):
    total_num = 0
    for p in list(model.parameters()):
        num = p.shape + (2,) if p.is_complex() else p.shape
        total_num = total_num + math.prod(num)
    return total_num
