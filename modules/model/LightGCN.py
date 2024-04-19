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
from ..utils import (inner_product, l2_loss)

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
        #self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        self.f = nn.Sigmoid()

    def forward(self, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj, self.ent_embeddings.weight)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items - self.num_users, item_embeddings)
        neg_item_embs = F.embedding(neg_items - self.num_users, item_embeddings)

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))

        # Reg Loss
        reg_loss = l2_loss(
            self.ent_embeddings(users),
            self.ent_embeddings(items),
            self.ent_embeddings(neg_items)
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
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj, self.ent_embeddings.weight)

    def getUsersRating(self, users):
        users_emb = self.ent_embeddings(users)
        items_emb = self.ent_embeddings.weight[self.num_users:]
        rating = self.f(torch.matmul(users_emb, items_emb.T))
        return rating
