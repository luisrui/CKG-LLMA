import pandas as pd
import numpy as np
import random
import torch 
import torch_geometric
from torch_geometric.data import Data
import os
import json
from reckit import randint_choice
from collections import (defaultdict, Iterable, OrderedDict)

class KGRecDataset(torch_geometric.data.Dataset):
    '''
    The dataset constructs graph nodes for background knowledge search and subgraph sampler.
    '''
    def __init__(self, args, num_user, num_item, transform = None, pre_transform = None):
        
        print('Loading data...')
        self.name = args['data']['name']
        self._data_dir = f'./dataset/{args['data']['name']}/'

        #self.triples = pd.read_csv(os.path.join(self._data_dir, 'triples.csv')) # Load the triples in three rows('head', 'relation', 'tail')
        #self.ent2id = json.load(open(os.path.join(self._data_dir, 'entity2id.json')))
        # self.rel2id = json.load(open(os.path.join(self._data_dir, 'relation2id.json')))

        super(KGRecDataset, self).__init__(self._data_dir, transform, pre_transform)
        
        self.num_user = num_user
        self.num_item = num_item

        self.struc_dataset = self.get(0)

    def __len__(self):
        return len(self.trainset)
    
    @property
    def raw_file_names(self):
        return ['triples.csv', 'entity2id.json']
    
    @property
    def processed_file_names(self):
        return [f'{self.name}.pt']

    @property
    def num_nodes(self):
        return self.get(0).num_nodes

    def strc_dataset(self):
        return self.struc_dataset
    
    def download(self):
        pass
    
    def process(self):
        '''
        Process the raw data into graph data with adjacency matrix and related egde information.
        '''
        triples = pd.read_csv(os.path.join(self._data_dir, 'triples.csv')) # Load the triples in three rows('head', 'relation', 'tail')
        ent2id = json.load(open(os.path.join(self._data_dir, 'entity2id.json')))

        edge_index = torch.tensor([[h, t] for h, t in zip(triples['head'], triples['tail'])], dtype=torch.long).t().contiguous()
        edge_type = torch.tensor([r for r in triples['relation']], dtype=torch.long)

        data = Data(edge_index = edge_index, edge_type = edge_type, num_nodes = len(ent2id))
        torch.save((data, None), self.processed_paths[0])
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))[0]
        return data
    
class Sampler(object):
    """Base class for all sampler to sample negative items.
    """
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


# class NegativeSampler(Sampler):
#     '''
#     Negative sampler to sample negative items for each user. 
#     '''
#     def __init__(self, dataset: KGRecDataset, num_neg=1, batch_size=1024, shuffle=True, drop_last=False):
        
#         super(NegativeSampler, self).__init__()
#         if num_neg <= 0:
#             raise ValueError("'num_neg' must be a positive integer.")

#         self.num_neg = num_neg
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.drop_last = drop_last
    
#     def __iter__(self):