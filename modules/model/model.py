import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
import torch.sparse as torch_sp
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
import json

from ..data import data_config, NegativeSampler
from ..utils import (inner_product, l2_loss, MarginLoss)

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def load_checkpoint(self, path, device):
        self.load_state_dict(torch.load(os.path.join(path), map_location=device))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
    
    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def get_parameters(self, mode = "numpy", param_dict = None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()    

class Model(BasicModel):
    def __init__(self, args : dict, norm_adj, kg, ent2id : dict, rel2id : dict, device : str):
        super(Model, self).__init__()
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.device = device

        self.n_layers = args['n_layers_lightgcn']
        self.kge_weight = args['loss_kge_weight']
        self.reg_weight = args['loss_reg_weight']

        # users, items, features
        self.num_users = data_config[args["data"]["name"]]['num_users']
        self.num_items = data_config[args["data"]["name"]]['num_items']
        self.ent_embeddings_kge = nn.Embedding(num_embeddings=len(ent2id), embedding_dim=args['embedding_dim'])
        self.ui_embeddings = nn.Embedding(num_embeddings=self.num_users + self.num_items, embedding_dim=args['embedding_dim'])
        #self.ent_embeddings_llm = nn.Embedding(num_embeddings=len(ent2id), embedding_dim=args['embedding_dim'])
        self.rel_embeddings = nn.Embedding(num_embeddings=len(rel2id), embedding_dim=args['embedding_dim'])

        self.fused_embeddings = nn.Embedding(num_embeddings=self.num_users + self.num_items, embedding_dim=args['embedding_dim'])
        # if args['isPretrain'] == 0:
        #     nn.init.normal_(self.ent_embeddings.weight, std=0.1)
        #     nn.init.normal_(self.rel_embedding, std=0.1)
        # else:
        #     self.ent_embeddings.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
        #     self.rel_embeddings.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
        #     print('use pretarined data')

        self.norm_adj = norm_adj.to(device)

        self.Neg_Sampler = NegativeSampler(args['data']['name'], kg, ent2id, rel2id, args['kg_neg_size'])
        self.KGEmodel = TransE()
        self.KGEloss = MarginLoss(margin=3.0)
        self.conv_gcn1 = RGCNConv(in_channels=args['embedding_dim'], out_channels=args['hidden_embedding_dim'], num_relations=len(rel2id), num_bases=len(rel2id))
        self.conv_gcn2 = RGCNConv(in_channels=args['hidden_embedding_dim'], out_channels=args['embedding_dim'], num_relations=len(rel2id), num_bases=len(rel2id))
        self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.remap_layer = nn.Linear(args['embedding_dim'] * 3, args['embedding_dim'])
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(args['embedding_dim'] * 2, args['embedding_dim']),
            nn.ReLU(),
            nn.Linear(args['embedding_dim'], args['embedding_dim'])
        )
        #### Evaluation
        self.rate_act_fn = nn.Sigmoid()
    
    def forward(self, edge_indexs, edge_types, users, items, neg_items):
        ### KG Embedding Learning(only training the ent_embeddings_kge, and rel_embeddings)
        x_list = []
        for edge_index, edge_type in zip(edge_indexs, edge_types):
            edge_index = edge_index.to(self.device)
            edge_type = edge_type.to(self.device)
            x = self.conv_gcn1(self.ent_embeddings_kge.weight, edge_index, edge_type)
            x = self.act_func(x)
            x = self.conv_gcn2(x, edge_index, edge_type)
            x_list.append(x)

        #Concatenation and Projection
        x = torch.concat(x_list, dim=-1)
        x = F.normalize(x, p=2, dim=1)
        x = self.remap_layer(x)

        merged_edge_index = torch.cat(edge_indexs, dim=1)
        merged_edge_type = torch.cat(edge_types)
        edge_index_expand, edge_type_expand = self.Neg_Sampler.Triples_neg_sample(merged_edge_index, merged_edge_type)
        edge_index_expand = edge_index_expand.to(self.device)
        edge_type_expand = edge_type_expand.to(self.device)

        head = x[edge_index_expand[0].long()]
        tail = x[edge_index_expand[1].long()]
        rels = self.rel_embeddings(edge_type_expand)

        pos_score, neg_score = self.KGEmodel(head, tail, rels, len(merged_edge_type))
        kge_loss = self.KGEloss(pos_score, neg_score)

        ### Rate Prediction Training(only training the ui_embeddings)
        user_embeddings, item_embeddings = self._forward_lightgcn(self.norm_adj, self.ui_embeddings.weight)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items - self.num_users, item_embeddings)
        neg_item_embs = F.embedding(neg_items - self.num_users, item_embeddings)

        ent_user_embs = self.ent_embeddings_kge(users)
        ent_pos_item_embs = self.ent_embeddings_kge(items)
        ent_neg_item_embs = self.ent_embeddings_kge(neg_items)

        fused_user_embs = self._fusion(ent_user_embs, user_embs)
        fused_pos_item_embs = self._fusion(ent_pos_item_embs, item_embs)
        fused_neg_item_embs = self._fusion(ent_neg_item_embs, neg_item_embs)

        sup_pos_ratings = inner_product(fused_user_embs, fused_pos_item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(fused_user_embs, fused_neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))

        # Reg Loss
        reg_loss = l2_loss(
            fused_user_embs,
            fused_pos_item_embs,
            fused_neg_item_embs
        )
        
        total_loss = bpr_loss + self.reg_weight * reg_loss + self.kge_weight * kge_loss

        self.fused_embeddings.weight.data[users] = fused_user_embs
        self.fused_embeddings.weight.data[items] = fused_pos_item_embs

        return total_loss, bpr_loss, kge_loss
 
    def _forward_lightgcn(self, norm_adj, ego_embeddings):
        '''
        Forward pass of LightGCN(Learning the ui embeddings)
        '''
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings
    
    def _fusion(self, kge_embeddings, ui_embeddings):
        ### MLP Fusion
        concat_embeddings = torch.cat((kge_embeddings, ui_embeddings), dim=-1)
        fused_embeddings = self.fusion_mlp(concat_embeddings)
        return fused_embeddings
    
    def getUsersRating(self, users):
        users_emb = self.fused_embeddings(users)
        items_emb = self.fused_embeddings.weight[self.num_users:]
        rating = self.rate_act_fn(torch.matmul(users_emb, items_emb.T))
        return rating

class TransE(nn.Module):
    '''
    TransE Model

    Compute the score of the triplets based on:
        score = ||h + r - t||_p
    '''
    def __init__(self, score_norm_flag : bool = False, p_norm : int = 1):
        super(TransE, self).__init__()
        self.score_norm_flag = score_norm_flag
        self.p_norm = p_norm

    def forward(self, head, tail, rel, pos_num, mode='normal'):
        if self.score_norm_flag:
            head = F.normalize(head, 2, -1)
            rel = F.normalize(rel, 2, -1)
            tail = F.normalize(tail, 2, -1)
        if mode != 'normal':
            head = head.view(-1, rel.shape[0], head.shape[-1])
            tail = tail.view(-1, rel.shape[0], tail.shape[-1])
            rel = rel.view(-1, rel.shape[0], rel.shape[-1])
        if mode == 'head_batch':
            score = head + (rel - tail)
        else:
            score = (head + rel) - tail
        score = torch.norm(score, self.p_norm, -1).flatten()
        pos_score = self._get_positive_score(score, pos_num)
        neg_score = self._get_negative_score(score, pos_num)
        return pos_score, neg_score
    
    def _get_positive_score(self, score, num_pos_samples):
        positive_score = score[:num_pos_samples]
        positive_score = positive_score.view(-1, num_pos_samples).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score, num_pos_samples):
        negative_score = score[num_pos_samples:]
        negative_score = negative_score.view(-1, num_pos_samples).permute(1, 0)
        return negative_score