import torch
from torch import nn as nn
import random
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np
import time

from .dataset import KGRecDataset, RecTrainDataset

class Extractor():
    '''
    Extracting subgraphs based on one-hop neighbours. When the core node is user, there should be two subgraphs:(user->user, user->item); when the core node is item, 
    there should be two subgraphs:(item->item, user->item). The total number of graphs of one pair of (user, item) should be 4.
    '''
    def __init__(self, args : dict, srcKG : KGRecDataset, recData: RecTrainDataset):

        print('Initializing Subgraph Extractor...')
        self.max_neighbors = args['max_sample_neighbors']
        self.name = args['data']['name']

        self.ent2id = args['ent2id']
        self.rel2id = args['rel2id']
        # self.id2ent = {Id : ent for ent, Id in zip(ent2id.keys(), ent2id.values())}
        # self.id2rel = {Id : rel for rel, Id in zip(rel2id.keys(), rel2id.values())}
        self.batch_size = args['extract_batch_size']

        self.num_user = args['num_users']
        self.num_items = args['num_items']
        
        self.u_of_i = recData.u_of_i
        self.u_of_u = srcKG.u_of_u
        self.i_of_i = srcKG.i_of_i
        self.i_of_a = srcKG.i_of_a
        # self.i_of_u = srcKG.i_of_u

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
        #batch_items = batch_pos_items

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
                u_u_extend_info = [tuple([user.item(), self.rel2id['co-liked'], co_usr.item()]) for user, nbrs in zip(batch_users, sampled_user_neighbors) for co_usr in nbrs if co_usr != 0]
                subgraph_uu.extend(u_u_extend_info)
                
                selected_co_items = list()
                for user, _, co_usr in u_u_extend_info:
                    common_liked_items = list(set(self.u_of_i[user]) & set(self.u_of_i[co_usr]) & set(batch_items.tolist()))
                    if common_liked_items:
                        for i in range(len(common_liked_items)):
                            co_item = common_liked_items[i]
                            if co_item not in selected_co_items:
                                selected_co_items.append(co_item)
                                subgraph_uu.append(tuple([user, self.rel2id['liked'], co_item]))
                                subgraph_uu.append(tuple([co_usr, self.rel2id['liked'], co_item]))
                                subgraph_uu.extend([tuple([co_item, r, t]) for r, t in self.i_of_a[co_item]])
                                break
                
                subgraph_uu = set(subgraph_uu)
                subgraph_uu = [list(triple) for triple in subgraph_uu]

            elif aug_tupe == 'ui':
                # User -> Item Subgraph (only consider positive items)
                selected_items = set()
                subgraph_ui.extend([tuple([user.item(), self.rel2id['liked'], item.item()]) for user, item in zip(batch_users, batch_pos_items)])
                subgraph_ui.extend([tuple([item.item(), r, t]) for item in batch_pos_items for r, t in self.i_of_a[item.item()]])
                selected_items.update(batch_pos_items.tolist())

                pos_item_neighbors = [list(set(self.u_of_i[user.item()]) & (set(batch_items.tolist()) - selected_items)) for user in batch_users]
                pos_item_neighbors_lens = [len(nbrs) for nbrs in pos_item_neighbors]
                max_len = max(pos_item_neighbors_lens)
                padded_pos_item_neighbors = [nbrs + [0] * (max_len - len(nbrs)) for nbrs in pos_item_neighbors]
                padded_pos_item_neighbors = torch.tensor(padded_pos_item_neighbors, dtype=torch.long)
                rand_idx = torch.rand(padded_pos_item_neighbors.shape).argsort(dim=1)
                top_k_idx = rand_idx[:, :self.max_neighbors]
                sampled_pos_item_neighbors = torch.gather(padded_pos_item_neighbors, 1, top_k_idx)
                u_i_extend_info = [tuple([user.item(), self.rel2id['liked'], ex_item.item()]) for user, nbrs in zip(batch_users, sampled_pos_item_neighbors) for ex_item in nbrs if ex_item != 0]
                subgraph_ui.extend(u_i_extend_info)
                subgraph_ui.extend([tuple([ex_item, r, t]) for _, _, ex_item in u_i_extend_info for r, t in self.i_of_a[ex_item]])
                
                subgraph_ui = set(subgraph_ui)
                subgraph_ui = [list(triple) for triple in subgraph_ui]
                            
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

                subgraph_ii.extend([tuple([item.item(), r, t]) for item in batch_items for r, t in self.i_of_a[item.item()]])
                i_i_extend_info = [tuple([item.item(), rel.item(), co_item.item()]) for item, nbrs in zip(batch_items, sampled_item_neighbors) for rel, co_item in nbrs if co_item != 0]
                subgraph_ii.extend(i_i_extend_info)
                subgraph_ii.extend([tuple([co_item, r, t]) for _, _, co_item in i_i_extend_info for r, t in self.i_of_a[co_item]])

                subgraph_ii = set(subgraph_ii)
                subgraph_ii = [list(triple) for triple in subgraph_ii]

            else:
                raise ValueError(f'Invalid type {aug_tupe} for subgraph extraction')

        return [subgraph_uu, subgraph_ui, subgraph_ii]

    def sample_subgraph_origin(self, 
                        aug_types : list, 
                        batch_users: torch.Tensor, 
                        batch_pos_items: torch.Tensor, 
                        batch_neg_items: torch.Tensor):
        
        subgraph_ui, subgraph_ii = [], []
        #batch_items = torch.cat((batch_pos_items, batch_neg_items))
        batch_items = batch_pos_items

        for aug_tupe in aug_types:
            start_time = time.time()
            if aug_tupe == 'ui':
                # User -> Item Subgraph (only consider positive items)
                selected_items = set()
                subgraph_ui.extend([tuple([user.item(), self.rel2id['liked'], item.item()]) for user, item in zip(batch_users, batch_pos_items)])
                subgraph_ui.extend([tuple([item.item(), r, t]) for item in batch_pos_items for r, t in self.i_of_a[item.item()]])
                selected_items.update(batch_pos_items.tolist())

                # pos_item_neighbors = [list(set(self.u_of_i[user.item()]) & (set(batch_items.tolist()) - selected_items)) for user in batch_users]
                # pos_item_neighbors_lens = [len(nbrs) for nbrs in pos_item_neighbors]
                # max_len = max(pos_item_neighbors_lens)
                # padded_pos_item_neighbors = [nbrs + [0] * (max_len - len(nbrs)) for nbrs in pos_item_neighbors]
                # padded_pos_item_neighbors = torch.tensor(padded_pos_item_neighbors, dtype=torch.long)
                # rand_idx = torch.rand(padded_pos_item_neighbors.shape).argsort(dim=1)
                # top_k_idx = rand_idx[:, :self.max_neighbors]
                # sampled_pos_item_neighbors = torch.gather(padded_pos_item_neighbors, 1, top_k_idx)
                # u_i_extend_info = [tuple([user.item(), self.rel2id['liked'], ex_item.item()]) for user, nbrs in zip(batch_users, sampled_pos_item_neighbors) for ex_item in nbrs if ex_item != 0]
                # subgraph_ui.extend(u_i_extend_info)
                # subgraph_ui.extend([tuple([ex_item, r, t]) for _, _, ex_item in u_i_extend_info for r, t in self.i_of_a[ex_item]])
                
                subgraph_ui = set(subgraph_ui)
                subgraph_ui = [list(triple) for triple in subgraph_ui]
                end_time = time.time() 
                print(f'UI time: {end_time - start_time}')
                start_time = time.time()            
            elif aug_tupe == 'ii':
                 # Item -> Item Subgraph
                # item_neighbors = [[(rel, nbr) for rel, nbr in self.i_of_i[item.item()] if nbr in batch_items] for item in batch_items]
                # item_neighbors_lens = [len(nbrs) for nbrs in item_neighbors]
                # max_len = max(item_neighbors_lens)
                # padded_item_neighbors = [nbrs + [(0, 0)] * (max_len - len(nbrs)) for nbrs in item_neighbors]
                # padded_item_neighbors = torch.tensor(padded_item_neighbors, dtype=torch.long)
                # rand_idx = torch.rand(padded_item_neighbors.shape[:2]).argsort(dim=1)
                # top_k_idx = rand_idx[:, :self.max_neighbors]
                # batch_indices = torch.arange(len(batch_items)).unsqueeze(1).expand(-1, self.max_neighbors)
                # sampled_item_neighbors = padded_item_neighbors[batch_indices, top_k_idx]

                # subgraph_ii.extend([tuple([item.item(), r, t]) for item in batch_items for r, t in self.i_of_a[item.item()]])
                # i_i_extend_info = [tuple([item.item(), rel.item(), co_item.item()]) for item, nbrs in zip(batch_items, sampled_item_neighbors) for rel, co_item in nbrs if co_item != 0]
                # subgraph_ii.extend(i_i_extend_info)
                # subgraph_ii.extend([tuple([co_item, r, t]) for _, _, co_item in i_i_extend_info for r, t in self.i_of_a[co_item]])

                # subgraph_ii = set(subgraph_ii)
                # subgraph_ii = [list(triple) for triple in subgraph_ii]
                unique_items = set(batch_items.tolist())
                tem_relations = {item: set((rel, tail) for rel, tail in self.i_of_i[item] if tail in unique_items) 
                              for item in unique_items}
                # Add attributes for each unique item
                subgraph_ii.extend([tuple([item, r, t]) for item in unique_items for r, t in self.i_of_a[item]])
                target_add_relations = np.array([tuple([item, rel, tail]) for item in unique_items for rel, tail in tem_relations[item]])
                keep_indices = np.random.choice(len(target_add_relations), len(target_add_relations)//10, replace=False)
                target_add_relations = target_add_relations[keep_indices]
                target_add_relations = [tuple(triple) for triple in target_add_relations]
                # Find relations between items in the batch
                subgraph_ii.extend(target_add_relations)
                # Remove duplicates (if any) and convert to list of lists
                subgraph_ii = set(subgraph_ii)
                subgraph_ii = [list(triple) for triple in set(tuple(t) for t in subgraph_ii)]
                
                end_time = time.time()
                print(f'II time: {end_time - start_time}')

            else:
                raise ValueError(f'Invalid type {aug_tupe} for subgraph extraction')

        return [subgraph_ui, subgraph_ii]


        