import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
import torch.sparse as torch_sp
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
import json

from .model import BasicModel
from ..utils import (inner_product, EmbLoss, BPRLoss, xavier_uniform_initialization)

class LightGCN(BasicModel):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers, batch_size):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.reg_loss_weight = 1 / batch_size

        # self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        # self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.ent_embeddings = nn.Embedding(self.num_users + self.num_items, self.embed_dim)
        #self.ent_embeddings.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
        nn.init.normal_(self.ent_embeddings.weight, std=0.1)
        self.regloss = EmbLoss()
        self.bprloss = BPRLoss()
        #self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None
        
        self.apply(xavier_uniform_initialization)
        self.f = nn.Sigmoid()

        # pre_embs = np.load('pretrained_embs_lightgcn.npz')
        # pre_users_emb = torch.Tensor(pre_embs['uembs'])
        # pre_items_emb = torch.Tensor(pre_embs['iembs'])
        # self.user_embbeddings = nn.Embedding.from_pretrained(pre_users_emb, freeze=True)
        # self.item_embbeddings = nn.Embedding.from_pretrained(pre_items_emb, freeze=True)

    def forward(self, users, items, neg_items):
        all_embeddings = self._forward_gcn(self.norm_adj, self.ent_embeddings.weight)

        user_embs = all_embeddings[users]
        item_embs = all_embeddings[items]
        neg_item_embs = all_embeddings[neg_items]

        sup_pos_ratings = torch.mul(user_embs, item_embs).sum(dim=1)
        sup_neg_ratings = torch.mul(user_embs, neg_item_embs).sum(dim=1)

        bpr_loss = self.bprloss(sup_pos_ratings, sup_neg_ratings)

        # Reg Loss
        reg_loss = self.regloss(
            self.ent_embeddings(users),
            self.ent_embeddings(items),
            self.ent_embeddings(neg_items),
            require_pow=True
        )

        loss = bpr_loss + self.reg_loss_weight * reg_loss
        return loss, bpr_loss
    
    def _forward_gcn(self, norm_adj, x):
        all_embeddings = [x]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                x = torch_sp.mm(norm_adj[k], x)
            else:
                x = torch_sp.mm(norm_adj, x)
            all_embeddings += [x]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        # user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return all_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(LightGCN, self).eval()
        #self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj, self.ent_embeddings.weight)

    def getUsersRating(self, users):
        all_embeddings = self._forward_gcn(self.norm_adj, self.ent_embeddings.weight)
        users_emb = all_embeddings[users]
        items_emb = all_embeddings[self.num_users:]
        rating = torch.matmul(users_emb, items_emb.T) 
        #rating = self.f(torch.matmul(users_emb, items_emb.T))
        return rating
    
    def getPretrainedRating(self, users):
        # all_embeddings =  torch.cat([self.user_embbeddings.weight, self.item_embbeddings.weight], dim=0)
        # embeddings_list = [all_embeddings]

        # for layer_idx in range(self.n_layers):
        #     all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
        #     embeddings_list.append(all_embeddings)
        # lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        # lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        # user_all_embeddings, item_all_embeddings = torch.split(
        #     lightgcn_all_embeddings, [self.n_users, self.n_items]
        # )

        # users_emb = user_all_embeddings[users]
        # items_emb = item_all_embeddings[1:]
        
        users_emb = self.user_embbeddings(users)
        items_emb = self.item_embbeddings.weight[1:]
        rating = torch.matmul(users_emb, items_emb.transpose(0, 1))
        return rating
