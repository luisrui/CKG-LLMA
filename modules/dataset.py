import pandas as pd
import numpy as np
import random
import torch 
import torch_geometric
from torch_geometric.data import Data
import os
import json
from collections import (defaultdict, Iterable, OrderedDict)
from tqdm import tqdm

config = {
    'AmazonBook' : {
        'user' : 'User_id',
        'item' : 'Id',
        'review' : 'review/text',
        'num_users' : 836500,
        'num_items' : 200880, 
    },
    'AmazonBookTiny' : {
        'user' : 'User_id',
        'item' : 'Id',
        'review' : 'review/text',
        'num_users' : 12634,
        'num_items' : 12568, 
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
        self.ent2id = json.load(open(os.path.join(self._data_dir, 'entity2id.json')))
        self.rel2id = json.load(open(os.path.join(self._data_dir, 'relation2id.json')))
        self.num_user = config[self.name]['num_users']
        self.num_items = config[self.name]['num_items']

        self.i_of_u = defaultdict(list) ## item-user with Liked relation
        self.u_of_i = defaultdict(list) ## user-item with Liked relation
        self.u_of_u = defaultdict(list) ## user-user with Co-liked relation
        self.i_of_a = defaultdict(list) ## item-attribute with Has attr relations
        self.i_of_i = defaultdict(list) ## item-item with Co-attr relations

        super(KGRecDataset, self).__init__(self._data_dir, transform, pre_transform)

        if os.path.exists(f'./dataset/{self.name}/pre_saved/u_of_u.pt'):
            self._load_adj()
        else:
            raise ValueError('No pre-saved u-i relations found. Please run the pre-processing script first.')

        self.struc_dataset = self.get(0)

    def __len__(self):
        return len(self.trainset)
    
    def _load_adj(self):
        self.u_of_i = torch.load(f'./dataset/{self.name}/pre_saved/u_of_i.pt')
        self.u_of_u = torch.load(f'./dataset/{self.name}/pre_saved/u_of_u.pt')
        self.i_of_i = torch.load(f'./dataset/{self.name}/pre_saved/i_of_i.pt')
        self.i_of_a = torch.load(f'./dataset/{self.name}/pre_saved/i_of_a.pt')
        self.i_of_u = torch.load(f'./dataset/{self.name}/pre_saved/i_of_u.pt')

    @property
    def raw_file_names(self):
        return ['triples.csv', 'entity2id.json']
    
    @property
    def processed_file_names(self):
        return [f'{self.name}.pt']

    @property
    def num_nodes(self):
        return self.get(0).num_nodes

    def get_struc_dataset(self):
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

        ## Build self.u_of_i and self.i_of_u
        print('Building user-item and item-user relations...')
        user_item = torch.zeros((self.num_user, self.num_items), dtype=torch.bool)
        item_user = torch.zeros((self.num_items, self.num_user), dtype=torch.bool)
        liked_mask = (edge_type == self.rel2id['liked']) & (edge_index[0] < self.num_user) & \
            (self.num_user <= edge_index[1]) & (edge_index[1] < self.num_user + self.num_items)
        user_item[edge_index[0][liked_mask], edge_index[1][liked_mask] - self.num_user] = True
        self.u_of_i = {u: (np.flatnonzero(row) + self.num_user).tolist() for u, row in enumerate(user_item.numpy())}
        item_user[edge_index[0][liked_mask], edge_index[1][liked_mask] - self.num_user] = True
        self.i_of_u = {i + self.num_user: np.flatnonzero(row).tolist() for i, row in enumerate(item_user.numpy())}
        
        ## Build self.u_of_u
        print('Building user-user relations...')
        user_user = torch.zeros((self.num_user, self.num_user), dtype=torch.bool)
        coliked_mask = (edge_type == self.rel2id['co-liked']) & (edge_index[0] < self.num_user) & (edge_index[1] < self.num_user)
        user_user[edge_index[0][coliked_mask], edge_index[1][coliked_mask]] = True
        self.u_of_u = {u: np.flatnonzero(row).tolist() for u, row in enumerate(user_user.numpy())}
        
        # Build self.i_of_i
        print('Building item-item relations...')
        item_mask = (edge_index[0] >= self.num_user) & (edge_index[0] < self.num_user + self.num_items) & \
            (edge_index[1] >= self.num_user) & (edge_index[1] < self.num_user + self.num_items)
        #item_item = torch.full((self.num_items, self.num_items), -1, dtype=torch.int64)
        item_hs, item_ts, ii_rels = edge_index[0][item_mask].numpy(), edge_index[1][item_mask].numpy(), edge_type[item_mask].numpy()
        for i in tqdm(range(len(item_hs))):
            h, t, r = item_hs[i], item_ts[i], ii_rels[i]
            self.i_of_i[h].append((r, t))
            self.i_of_i[t].append((r, h))
        #self.i_of_i = {i : [(h, r, t) for h, r, t in zip(edge_index[0][item_mask], edge_type[item_mask], edge_index[1][item_mask]) if i == h or i == t] for i in range(self.num_user, self.num_user+self.num_items)}
        
        ## Build self.i_of_a
        print('Building item-attribute relations...')
        attr_mask = (edge_index[0] >= self.num_user) & (edge_index[0] < self.num_user+self.num_items) & (edge_index[1] >= self.num_user+self.num_items)
        target_items, target_attrs, target_rels = edge_index[0][attr_mask].numpy(), edge_index[1][attr_mask].numpy(), edge_type[attr_mask].numpy()
        for i in tqdm(range(len(target_items))):
            h, t, r = target_items[i], target_attrs[i], target_rels[i]
            self.i_of_a[h].append((r, t))
        # self.i_of_a = {i: [(i, r, t) for i, r, t in zip(edge_index[0][attr_mask], edge_type[attr_mask], edge_index[1][attr_mask]) if i == i] for i in range(self.num_user, self.num_user+self.num_items)}
            
        os.makedirs(f'./dataset/{self.name}/pre_saved/', exist_ok=True)
        torch.save(self.u_of_u, f'./dataset/{self.name}/pre_saved/u_of_u.pt')
        torch.save(self.i_of_i, f'./dataset/{self.name}/pre_saved/i_of_i.pt') 
        torch.save(self.u_of_i, f'./dataset/{self.name}/pre_saved/u_of_i.pt')
        torch.save(self.i_of_u, f'./dataset/{self.name}/pre_saved/i_of_u.pt')
        torch.save(self.i_of_a, f'./dataset/{self.name}/pre_saved/i_of_a.pt')
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))[0]
        return data
    
    def _gen_adj(self):
        edge_index = self.struc_dataset.edge_index
        edge_type = self.struc_dataset.edge_type
         ## Build self.u_of_i
        print('Building user-item relations...')
        user_item = torch.zeros((self.num_user, self.num_items), dtype=torch.bool)
        liked_mask = (edge_type == self.rel2id['liked']) & (edge_index[0] < self.num_user) & \
            (self.num_user <= edge_index[1]) & (edge_index[1] < self.num_user + self.num_items)
        user_item[edge_index[0][liked_mask], edge_index[1][liked_mask] - self.num_user] = True
        self.u_of_i = {u: (np.flatnonzero(row) + self.num_user).tolist() for u, row in enumerate(user_item.numpy())}
        
        ## Build self.u_of_u
        print('Building user-user relations...')
        user_user = torch.zeros((self.num_user, self.num_user), dtype=torch.bool)
        coliked_mask = (edge_type == self.rel2id['co-liked']) & (edge_index[0] < self.num_user) & (edge_index[1] < self.num_user)
        user_user[edge_index[0][coliked_mask], edge_index[1][coliked_mask]] = True
        self.u_of_u = {u: np.flatnonzero(row).tolist() for u, row in enumerate(user_user.numpy())}
        
        # Build self.i_of_i
        print('Building item-item relations...')
        item_mask = (edge_index[0] >= self.num_user) & (edge_index[0] < self.num_user + self.num_items) & \
            (edge_index[1] >= self.num_user) & (edge_index[1] < self.num_user + self.num_items)
        #item_item = torch.full((self.num_items, self.num_items), -1, dtype=torch.int64)
        item_hs, item_ts, ii_rels = edge_index[0][item_mask].numpy(), edge_index[1][item_mask].numpy(), edge_type[item_mask].numpy()
        for i in tqdm(range(len(item_hs))):
            h, t, r = item_hs[i], item_ts[i], ii_rels[i]
            self.i_of_i[h].append((r, t))
            self.i_of_i[t].append((r, h))
        #self.i_of_i = {i : [(h, r, t) for h, r, t in zip(edge_index[0][item_mask], edge_type[item_mask], edge_index[1][item_mask]) if i == h or i == t] for i in range(self.num_user, self.num_user+self.num_items)}
        
        ## Build self.i_of_a
        print('Building item-attribute relations...')
        attr_mask = (edge_index[0] >= self.num_user) & (edge_index[0] < self.num_user+self.num_items) & (edge_index[1] >= self.num_user+self.num_items)
        target_items, target_attrs, target_rels = edge_index[0][attr_mask].numpy(), edge_index[1][attr_mask].numpy(), edge_type[attr_mask].numpy()
        for i in tqdm(range(len(target_items))):
            h, t, r = target_items[i], target_attrs[i], target_rels[i]
            self.i_of_a[h].append((r, t))
        # self.i_of_a = {i: [(i, r, t) for i, r, t in zip(edge_index[0][attr_mask], edge_type[attr_mask], edge_index[1][attr_mask]) if i == i] for i in range(self.num_user, self.num_user+self.num_items)}
            
        os.makedirs(f'./dataset/{self.name}/pre_saved/', exist_ok=True)
        torch.save(self.u_of_u, f'./dataset/{self.name}/pre_saved/u_of_u.pt')
        torch.save(self.i_of_i, f'./dataset/{self.name}/pre_saved/i_of_i.pt') 
        torch.save(self.u_of_i, f'./dataset/{self.name}/pre_saved/u_of_i.pt')
        #torch.save(self.i_of_u, f'./dataset/{self.name}/pre_saved/i_of_u.pt')
        torch.save(self.i_of_a, f'./dataset/{self.name}/pre_saved/i_of_a.pt')
    

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
        #self.i_of_u = defaultdict(list)
        if os.path.exists(f'./dataset/{name}/pre_saved/'):
            self._load_uoi()
        else:
            self._count_uoi(dataset[config[self.name]['user']], dataset[config[self.name]['item']])
    
    def _load_uoi(self):
        self.u_of_i = torch.load(f'./dataset/{self.name}/pre_saved/u_of_i.pt')
        #self.i_of_u = torch.load(f'./dataset/{self.name}/pre_saved/i_of_u.pt')

    def _count_uoi(self, users, items):
        for u, i in zip(users, items):
            u_id = self.e2id[u]
            i_id = self.e2id[i]
            self.u_of_i[u_id].append(i_id)
            #self.i_of_u[i_id].append(u_id)

        for u in self.u_of_i.keys():
            self.u_of_i[u] = np.array(list(set(self.u_of_i[u])))
        # for i in self.i_of_u.keys():
        #     self.i_of_u[i] = np.array(list(set(self.i_of_u[i])))
        
        os.makedirs(f'./dataset/{self.name}/pre_saved/', exist_ok=True)
        torch.save(self.u_of_i, f'./dataset/{self.name}/pre_saved/u_of_i.pt')
        #torch.save(self.i_of_u, f'./dataset/{self.name}/pre_saved/i_of_u.pt')
    
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
        user = user.item()
        mask = np.in1d(ar1=tmp, ar2=self.u_of_i[user], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg