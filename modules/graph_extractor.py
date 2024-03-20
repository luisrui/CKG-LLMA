import torch
from torch import nn as nn
import random
from .dataset import KGRecDataset
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np

class Extractor():
    '''
    Extracting subgraphs based on one-hop neighbours. When the core node is user, there should be two subgraphs:(user->user, user->item); when the core node is item, 
    there should be two subgraphs:(item->item, item->user). The total number of graphs of one pair of (user, item) should be 4.
    '''
    def __init__(self, args : dict, num_user : int, num_items : int, ent2id : list, rel2id : list, srcKG : KGRecDataset):

        print('Initializing Subgraph Extractor...')
        self.n_iter = args['n_iter']
        self.max_neighbors = args['max_sample_neighbors']
        self.name = args['data']['name']

        self.ent2id = ent2id
        self.rel2id = rel2id
        # self.id2ent = {Id : ent for ent, Id in zip(ent2id.keys(), ent2id.values())}
        # self.id2rel = {Id : rel for rel, Id in zip(rel2id.keys(), rel2id.values())}
        self.batch_size = args['batch_size']
        self.sample_size = args['negnum'] + 1

        self.num_user = num_user
        self.num_items = num_items
        
        self.u_of_i = srcKG.u_of_i
        self.u_of_u = srcKG.u_of_u
        self.i_of_i = srcKG.i_of_i
        self.i_of_a = srcKG.i_of_a
        self.i_of_u = srcKG.i_of_u

    def _get_neighbors(self, v):  # -> tuple[list, list]:
        """
        v is batch sized indices for items
        v: [batch_size, 1]
        """
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = (
                torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1))
            )
            neighbor_relations = (
                torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1))
            )
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations
    
    def sample_subgraph(self, aug_types : list, batch_users : torch.Tensor, batch_items : torch.Tensor):
        '''
        Extract subgraphs based on the given types. Subgraphs are constructed based on their triples. The user-user subgraph is designed to 
        '''
        target_types = ['uu', 'ui', 'iu', 'ii']
        graph_uu, graph_ui, graph_iu, graph_ii = [], [], [], []

        for ex_type in aug_types:
            if ex_type not in target_types:
                raise ValueError(f'Invalid type {ex_type} for subgraph extraction')
            
            if ex_type == 'uu': ## Co-liked relations
                subgraph_uus = []
                for i in range(self.batch_size):
                    user = batch_users[0, i].item()
                    user_neighbors = self.u_of_u[user]
                    subgraph_uu = []
                    if len(user_neighbors) >= self.max_neighbors:
                        sampled_nbrs = random.sample(population=user_neighbors, k=self.max_neighbors)
                        subgraph_uu = [[user, self.rel2id['co-liked'], co_usr] for co_usr in sampled_nbrs]
                        for co_usr in sampled_nbrs:
                            # Find an item that both users liked
                            common_liked_items = list(set(self.u_of_i[user]) & set(self.u_of_i[co_usr]))
                            if common_liked_items:
                                co_item = random.choice(common_liked_items)
                                subgraph_uu.append([user, self.rel2id['liked'], co_item])
                                subgraph_uu.extend([[co_item, r, t] for r, t in self.i_of_a[co_item]])
                    print(subgraph_uu)
                    subgraph_uus.append(subgraph_uu)
                graph_uu = [subgraph_uus for _ in self.sample_size]
            
            if ex_type == 'ui': ## User-Item relations
                subgraph_uis = []
                for i in range(self.batch_size):
                    user = batch_users[0, i].item()
                    item = batch_items[0, i].item()
                    subgraph_ui = [[user, self.rel2id['liked'], item]]
                    subgraph_ui.extend([[item, r, t] for r, t in self.i_of_a[item]])
                    item_neighbors = self.u_of_i[item]
                    if len(item_neighbors) >= self.max_neighbors:
                        sampled_nbrs = random.sample(population=item_neighbors, k=self.max_neighbors)
                        subgraph_ui.extend([[user, self.rel2id['liked'], co_item] for co_item in sampled_nbrs])
                    subgraph_uis.append(subgraph_ui)
                graph_ui = [subgraph_uis for _ in self.sample_size]
            
            if ex_type == 'ii':
                for i in range(self.sample_size):
                    subgraph_iis = []
                    for j in range(self.batch_size):
                        item = batch_items[i, j].item()
                        rel_nbritems = self.i_of_i[item]
                        subgraph_ii = [[item, r, t] for r, t in self.i_of_a[item]]
                        if len(rel_nbritems) >= self.max_neighbors:
                            sampled_nbrs = random.sample(population=item_neighbors, k=self.max_neighbors)
                            subgraph_ii.extend([[item, rel, co_item] for rel, co_item in rel_nbritems])
                            subgraph_ii.extend([[co_item, r, t] for r, t in self.i_of_a[co_item] for _, co_item in rel_nbritems])
                        subgraph_iis.append(subgraph_ii)
                    graph_ii.append(subgraph_iis)
            
            if ex_type == 'iu':
                subgraph_ius = []
                for i in range(self.batch_size):
                    item = batch_items[0, i].item()
                    user_neighbors = self.i_of_u[user]
                    if len(user_neighbors) >= self.max_neighbors:
                        nbr_users = random.sample(population=user_neighbors, k=self.max_neighbors)
                        subgraph_iu.extend([[co_user, self.rel2id['liked'], item] for co_user in nbr_users])
                    else:
                        subgraph_iu = [[co_user, self.rel2id['liked'], item] for co_user in user_neighbors]
                    subgraph_iu.extend([[item, r, t] for r, t in self.i_of_a[item]])
                    subgraph_ius.append(subgraph_iu)
                graph_iu = [subgraph_ius for _ in self.sample_size]

        return [graph_uu, graph_ui, graph_iu, graph_ii]
