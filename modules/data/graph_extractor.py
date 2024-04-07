import torch
from torch import nn as nn
import random
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np

from .dataset import KGRecDataset

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

    # def sample_subgraph(self, aug_types : list, batch_users : torch.Tensor, batch_items : torch.Tensor):
    #     '''
    #     Extract subgraphs based on the given types. Subgraphs are constructed based on their triples.
    #     '''
    #     target_types = ['uu', 'ui', 'iu', 'ii']
    #     subgraph_uus, subgraph_uis, subgraph_ius, subgraph_iis = [], [], [], []

    #     user_item_pairs = np.array(np.meshgrid(batch_users, batch_items)).T.reshape(-1, 2)

    #     for aug_tupe in aug_types:
    #         if aug_tupe not in target_types:
    #             raise ValueError(f'Invalid type {aug_tupe} for subgraph extraction')

    #         if aug_tupe == 'uu':
    #             for user, _ in user_item_pairs:
    #                 user_neighbors = self.u_of_u[user]
    #                 sampled_nbrs = np.random.choice(user_neighbors, size=min(len(user_neighbors), self.max_neighbors), replace=False)
    #                 subgraph_uu = [[user, self.rel2id['co-liked'], co_usr] for co_usr in sampled_nbrs]
    #                 for co_usr in sampled_nbrs:
    #                     common_liked_items = list(set(self.u_of_i[user]) & set(self.u_of_i[co_usr]))
    #                     if common_liked_items:
    #                         co_item = np.random.choice(common_liked_items)
    #                         subgraph_uu.append([user, self.rel2id['liked'], co_item])
    #                         subgraph_uu.extend([[co_item, r, t] for r, t in self.i_of_a[co_item]])
    #                 subgraph_uus.append(subgraph_uu)

    #         if aug_tupe == 'ui':
    #             for user, item in user_item_pairs:
    #                 subgraph_ui = [[user, self.rel2id['liked'], item]]
    #                 subgraph_ui.extend([[item, r, t] for r, t in self.i_of_a[item]])
    #                 item_neighbors = self.u_of_i[item]
    #                 if len(item_neighbors) >= self.max_neighbors:
    #                     sampled_nbrs = np.random.choice(item_neighbors, size=self.max_neighbors, replace=False)
    #                     subgraph_ui.extend([[user, self.rel2id['liked'], co_item] for co_item in sampled_nbrs])
    #                 else:
    #                     subgraph_ui.extend([[user, self.rel2id['liked'], co_item] for co_item in item_neighbors])
    #                 subgraph_uis.append(subgraph_ui)

    #         if aug_tupe == 'ii':
    #             for _, item in user_item_pairs:
    #                 rel_nbritems = self.i_of_i[item]
    #                 subgraph_ii = [[item, r, t] for r, t in self.i_of_a[item]]
    #                 if len(rel_nbritems) >= self.max_neighbors:
    #                     sampled_nbrs = np.random.choice(rel_nbritems, size=self.max_neighbors, replace=False)
    #                     subgraph_ii.extend([[item, rel, co_item] for rel, co_item in sampled_nbrs])
    #                     for rel, co_item in sampled_nbrs:
    #                         subgraph_ii.extend([[co_item, r, t] for r, t in self.i_of_a[co_item]])
    #                 else:
    #                     subgraph_ii.extend([[item, rel, co_item] for rel, co_item in rel_nbritems])
    #                     for rel, co_item in rel_nbritems:
    #                         subgraph_ii.extend([[co_item, r, t] for r, t in self.i_of_a[co_item]])
    #                 subgraph_iis.append(subgraph_ii)

    #         if aug_tupe == 'iu':
    #             for user, item in user_item_pairs:
    #                 user_neighbors = self.i_of_u[user]
    #                 subgraph_iu = [[item, r, t] for r, t in self.i_of_a[item]]
    #                 if len(user_neighbors) >= self.max_neighbors:
    #                     nbr_users = np.random.choice(user_neighbors, size=self.max_neighbors, replace=False)
    #                     subgraph_iu.extend([[co_user, self.rel2id['liked'], item] for co_user in nbr_users])
    #                 else:
    #                     subgraph_iu.extend([[co_user, self.rel2id['liked'], item] for co_user in user_neighbors])
    #                 subgraph_ius.append(subgraph_iu)

    #     return [subgraph_uus, subgraph_uis, subgraph_ius, subgraph_iis]
    
    def sample_subgraph(self, 
                        aug_types : list, 
                        batch_users: torch.Tensor, 
                        batch_pos_items: torch.Tensor, 
                        batch_neg_items: torch.Tensor):
        '''
        Extract subgraphs based on the given batch of users, positive items, and negative items.
        Subgraphs are constructed based on their triples.
        '''
        subgraph_uu, subgraph_ui, subgraph_ii = [], [], []
        batch_items = torch.cat((batch_pos_items, batch_neg_items))
        
        for aug_tupe in aug_types:
            
            if aug_tupe == 'uu':
                 # User -> User Subgraph
                user_neighbors = [list(set(self.u_of_u[user.item()]) & set(batch_users.tolist())) for user in batch_users]
                user_neighbors_lens = [len(nbrs) for nbrs in user_neighbors]
                max_len = max(user_neighbors_lens)
                padded_user_neighbors = [nbrs + [0] * (max_len - len(nbrs)) for nbrs in user_neighbors]
                padded_user_neighbors = torch.tensor(padded_user_neighbors, dtype=torch.long)
                rand_idx = torch.rand(padded_user_neighbors.shape).argsort(dim=1)
                top_k_idx = rand_idx[:, :self.max_neighbors]
                sampled_user_neighbors = torch.gather(padded_user_neighbors, 1, top_k_idx)
                u_u_extend_info = [[user.item(), self.rel2id['co-liked'], co_usr.item()] for user, nbrs in zip(batch_users, sampled_user_neighbors) for co_usr in nbrs if co_usr != 0]
                subgraph_uu.extend(u_u_extend_info)

                for user, _, co_usr in u_u_extend_info:
                    common_liked_items = list(set(self.u_of_i[user]) & set(self.u_of_i[co_usr]) & set(batch_items.tolist()))
                    if common_liked_items:
                        co_item = np.random.choice(common_liked_items)
                        subgraph_uu.append([user, self.rel2id['liked'], co_item])
                        subgraph_uu.append([co_usr, self.rel2id['liked'], co_item])
                        subgraph_uu.extend([[co_item, r, t] for r, t in self.i_of_a[co_item]])

            elif aug_tupe == 'ui':
                # User -> Item Subgraph (only consider positive items)
                subgraph_ui.extend([[user.item(), self.rel2id['liked'], item.item()] for user, item in zip(batch_users, batch_pos_items)])
                subgraph_ui.extend([[item.item(), r, t] for item in batch_pos_items for r, t in self.i_of_a[item.item()]])

                pos_item_neighbors = [list(set(self.u_of_i[user.item()]) & set(batch_users.tolist())) for user in batch_users]
                pos_item_neighbors_lens = [len(nbrs) for nbrs in pos_item_neighbors]
                max_len = max(pos_item_neighbors_lens)
                padded_pos_item_neighbors = [nbrs + [0] * (max_len - len(nbrs)) for nbrs in pos_item_neighbors]
                padded_pos_item_neighbors = torch.tensor(padded_pos_item_neighbors, dtype=torch.long)
                rand_idx = torch.rand(padded_pos_item_neighbors.shape).argsort(dim=1)
                top_k_idx = rand_idx[:, :self.max_neighbors]
                sampled_pos_item_neighbors = torch.gather(padded_pos_item_neighbors, 1, top_k_idx)
                u_i_extend_info = [[user.item(), self.rel2id['liked'], ex_item.item()] for user, nbrs in zip(batch_users, sampled_pos_item_neighbors) for ex_item in nbrs if ex_item != 0]
                subgraph_ui.extend(u_i_extend_info)
                subgraph_ui.extend([[ex_item, r, t] for _, _, ex_item in u_i_extend_info for r, t in self.i_of_a[ex_item]])
            
            elif aug_tupe == 'ii':
                 # Item -> Item Subgraph
                item_neighbors = [[(rel, nbr) for rel, nbr in self.i_of_i[item.item()] if nbr in batch_items] for item in batch_items]
                item_neighbors_lens = [len(nbrs) for nbrs in item_neighbors]
                max_len = max(item_neighbors_lens)
                padded_item_neighbors = [nbrs + [(0, 0)] * (max_len - len(nbrs)) for nbrs in item_neighbors]
                padded_item_neighbors = torch.tensor(padded_item_neighbors, dtype=torch.long)
                rand_idx = torch.rand(padded_item_neighbors.shape[:2]).argsort(dim=1)
                top_k_idx = rand_idx[:, :self.max_neighbors]
                batch_indices = torch.arange(len(batch_items)).unsqueeze(1).expand(-1, self.max_neighbors)
                sampled_item_neighbors = padded_item_neighbors[batch_indices, top_k_idx]

                subgraph_ii.extend([[item.item(), r, t] for item in batch_items for r, t in self.i_of_a[item.item()]])
                i_i_extend_info = [[item.item(), rel.item(), co_item.item()] for item, nbrs in zip(batch_items, sampled_item_neighbors) for rel, co_item in nbrs if co_item != 0]
                subgraph_ii.extend(i_i_extend_info)
                subgraph_ii.extend([[co_item, r, t] for _, _, co_item in i_i_extend_info for r, t in self.i_of_a[co_item]])

            else:
                raise ValueError(f'Invalid type {aug_tupe} for subgraph extraction')

        return [subgraph_uu, subgraph_ui, subgraph_ii]
        