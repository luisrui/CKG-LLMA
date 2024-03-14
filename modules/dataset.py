import pandas as pd
import numpy as np
import random
import torch 
import torch_geometric
from torch_geometric.data import Data
import os
import json
from collections import (defaultdict, Iterable, OrderedDict)

config = {
    'AmazonBook' : {
        'user' : 'User_id',
        'item' : 'Id',
        'review' : 'review/text',
        'num_users' : 836500,
        'num_items' : 200880, 
    },
    'AmazonBookSmall' : {
        'user' : 'User_id',
        'item' : 'Id',
        'review' : 'review/text',
        'num_users' : 836500,
        'num_items' : 200880, 
    },
    'Yelp' : {
        'user' : 'xxx',
        'item' : 'xx'
    }
}
class KGRecDataset(torch_geometric.data.Dataset):
    '''
    The dataset constructs graph nodes for background knowledge search and subgraph sampler.
    '''
    def __init__(self, args, transform = None, pre_transform = None):
        
        print('Loading background knowledge graph data...')
        self.name = args['data']['name']
        self._data_dir = f'./dataset/{args["data"]["name"]}/'

        #self.triples = pd.read_csv(os.path.join(self._data_dir, 'triples.csv')) # Load the triples in three rows('head', 'relation', 'tail')
        #self.ent2id = json.load(open(os.path.join(self._data_dir, 'entity2id.json')))
        # self.rel2id = json.load(open(os.path.join(self._data_dir, 'relation2id.json')))

        super(KGRecDataset, self).__init__(self._data_dir, transform, pre_transform)

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

class RecTrainDataset(torch.utils.data.Dataset):
    '''
    The class for recommendation dataset, containing user-item interactions, reviews and positive ratings
    '''
    def __init__(self, args):
        print('Loading recommendation dataset...')
        self.negnum = args['negnum']
        self.name = args['data']['name']
        self._data_dir = f'./dataset/{self.name}/'
        self.data = pd.read_csv(os.path.join(self._data_dir, 'data_all.csv'))
        self.trainset = pd.read_csv(os.path.join(self._data_dir, 'train.csv'))
        self.ent2id = json.load(open(os.path.join(self._data_dir, 'entity2id.json')))
        self.rel2id = json.load(open(os.path.join(self._data_dir, 'relation2id.json')))
        # self.validset = pd.read_csv(os.path.join(self._data_dir, 'valid.csv'))
        # self.testset = pd.read_csv(os.path.join(self._data_dir, 'test.csv'))
        self._sampler = NegativeSampler(self.name, self.data, self.ent2id, self.rel2id, self.negnum)

    def __len__(self):
        return len(self.trainset)
    
    def __getitem__(self, idx):
        user_id = self.ent2id[self.trainset.iloc[idx][config[self.name]['user']]]
        item_id = self.ent2id[self.trainset.iloc[idx][config[self.name]['item']]]
        #user_id_negsampled, item_id_negsampled = self._sampler.neg_sample_fn(user_id, item_id)
        review = self.trainset.iloc[idx][config[self.name]['review']]
        #return user_id_negsampled, item_id_negsampled, review
        return user_id, item_id, review

    def negative_sample(self, users:torch.Tensor, items:torch.Tensor):
        return self._sampler.neg_sample_fn(users, items)

class Sampler(object):
    """Base class for all sampler to sample negative items.
    """
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class NegativeSampler(Sampler):
    '''
    Negative sampler to sample negative items for each user. 
    '''
    def __init__(self, 
                 name : str,
                 dataset:pd.DataFrame, 
                 ent2id:dict, 
                 rel2id:dict, 
                 num_neg:int = 1
                 ):
        
        super(NegativeSampler, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.name = name
        self.e2id = ent2id
        self.r2id = rel2id
        self.num_neg = num_neg

        self.u_of_i = defaultdict(list)
        self.i_of_u = defaultdict(list)
        if os.path.exists(f'./dataset/{name}/pre_saved/'):
            self._load_uoi()
        else:
            self._count_uoi(dataset[config[self.name]['user']], dataset[config[self.name]['user']])
    
    def _load_uoi(self):
        self.u_of_i = torch.load(f'./dataset/{self.name}/pre_saved/u_of_i.pt')
        self.i_of_u = torch.load(f'./dataset/{self.name}/pre_saved/i_of_u.pt')

    def _count_uoi(self, users, items):
        for u, i in zip(users, items):
            u_id = self.e2id[u]
            i_id = self.e2id[i]
            self.u_of_i[u_id].append(i_id)
            self.i_of_u[i_id].append(u_id)

        for u in self.u_of_i.keys():
            self.u_of_i[u] = np.array(list(set(self.u_of_i[u])))
        for i in self.i_of_u.keys():
            self.i_of_u[i] = np.array(list(set(self.i_of_u[i])))
        
        os.makedirs(f'./dataset/{self.name}/pre_saved/', exist_ok=True)
        torch.save(self.u_of_i, f'./dataset/{self.name}/pre_saved/u_of_i.pt')
        torch.save(self.i_of_u, f'./dataset/{self.name}/pre_saved/i_of_u.pt')
    
    def neg_sample_fn(self, users, items):
        #len_triples = batch_h.__len__()
        batch_u_sample = np.repeat(users.view(-1, 1).cpu().numpy(), 1 + self.num_neg, axis = -1)
        batch_i_sample = np.repeat(items.view(-1, 1).cpu().numpy(), 1 + self.num_neg, axis = -1)
        for idx, (u, i) in enumerate(zip(users, items)):
            last = 1
            if self.num_neg > 0:
                neg_items = self._normal_batch(u)
                if len(neg_items) > 0:
                    batch_i_sample[idx][last:last + len(neg_items)] = neg_items
                    last += len(neg_items)
        batch_users = batch_u_sample.transpose()
        batch_items = batch_i_sample.transpose()

        batch_users = torch.tensor(np.array(batch_users), dtype=torch.int32)
        batch_items = torch.tensor(np.array(batch_items), dtype=torch.int32)
        return batch_users, batch_items
    
    def _normal_batch(self, user):
        neg_list_items = []
        neg_cur_size = 0
        while neg_cur_size < self.num_neg:
            neg_tmp_items = self._corrupt_tail(user, num_max = (self.num_neg - neg_cur_size) * 2)
            neg_list_items.append(neg_tmp_items)
            neg_cur_size += len(neg_tmp_items)
        if neg_list_items != []:
            neg_list_items = np.concatenate(neg_list_items)

        return neg_list_items[:self.num_neg]
    
    def _corrupt_tail(self, user, num_max = 1000):
        # try:
        #     tmp = torch.tensor(random.sample(node_list.cpu().numpy().tolist(), k=num_max))
        # except:
        #     tmp = torch.tensor(random.sample(node_list.cpu().numpy().tolist(), k=len(node_list)))
        tmp = torch.tensor(random.sample(range(config[self.name]['num_users'], config[self.name]['num_users'] + config[self.name]['num_items']), k=num_max))
        mask = np.in1d(ar1=tmp, ar2=self.u_of_i[user], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg