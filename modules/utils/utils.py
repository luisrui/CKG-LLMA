import yaml
import torch
import random
import numpy as np


def read_yaml(path):
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def set_random_seed(seed=2020):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def triples_transfer_to_graph(subgraphs: list):
    edge_index_list = []
    edge_type_list = []

    for subgraph in subgraphs:
        heads, rels, tails = zip(*subgraph)

        heads = np.array(heads)
        tails = np.array(tails)
        rels = np.array(rels)

        edge_index_sub = np.stack((heads, tails), axis=0)
        edge_index_sub = torch.from_numpy(edge_index_sub).long()
        edge_type_sub = torch.from_numpy(rels).long()

        edge_index_list.append(edge_index_sub)
        edge_type_list.append(edge_type_sub)

    return edge_index_list, edge_type_list

def inner_product(a, b):
    return torch.sum(a * b, dim=-1)

def sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 32)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), step=batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)