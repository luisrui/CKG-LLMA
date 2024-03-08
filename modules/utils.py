import yaml
import torch
import random
import numpy as np


def read_yaml(path):
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def _set_random_seed(seed=2020):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True