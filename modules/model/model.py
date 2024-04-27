import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
import torch.sparse as torch_sp
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np
import scipy.sparse as sp


from ..data import data_config, NegativeSampler
from ..utils import (inner_product, l2_loss, MarginLoss, edge_softmax)

from .BasicModel import BasicModel
from .KGEmb import *

class Aggregator(MessagePassing):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        # edge_attr:  [num_edges]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: origin features [num_edges, in_dim]
        # edge_attr: edge_attr [num_edges]
        msg = x_j * edge_attr.view(-1, 1)
        return msg

    def update(self, aggr_out, x):
        # aggr_out: aggregated info [num_nodes, in_dim]
        # x: all nodes embeddings [num_nodes, in_dim]
        if self.aggregator_type == 'gcn':
            out = self.activation(self.W(x + aggr_out))              # (num_nodes, out_dim)
        elif self.aggregator_type == 'graphsage':
            out = self.activation(self.W(torch.cat([x, aggr_out], dim=1)))   # (num_nodes, out_dim)
        elif self.aggregator_type == 'bi-interaction':
            out1 = self.activation(self.W1(x + aggr_out))            # (num_nodes, out_dim)
            out2 = self.activation(self.W2(x * aggr_out))            # (num_nodes, out_dim)
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out
    
class Model(BasicModel):
    def __init__(self, args : dict, norm_adj, kg, ent2id : dict, rel2id : dict, device : str):
        super(Model, self).__init__()
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.num_rel = len(rel2id)
        self.device = device

        self.n_layers = args['n_layers_lightgcn']
        self.kge_weight = args['loss_kge_weight']
        self.reg_weight = args['loss_reg_weight']
        self.num_ag_layers = len(args['conv_dim_list'])
        self.conv_dim_list = [args['ent_embedding_dim']] + args['conv_dim_list']
        self.mess_dropout = args['mess_dropout']
        self.aggregation_type = args['aggregation_type']

        # users, items, features
        self.num_users = data_config[args["data"]["name"]]['num_users']
        self.num_items = data_config[args["data"]["name"]]['num_items']
        self.ent_embeddings_kge = nn.Embedding(num_embeddings=len(ent2id), embedding_dim=args['ent_embedding_dim'])
        #self.ui_embeddings = nn.Embedding(num_embeddings=self.num_users + self.num_items, embedding_dim=args['embedding_dim'])
        #self.ent_embeddings_llm = nn.Embedding(num_embeddings=len(ent2id), embedding_dim=args['embedding_dim'])
        self.rel_embeddings = nn.Embedding(num_embeddings=self.num_rel, embedding_dim=args['rel_embedding_dim'])

        #self.fused_embeddings = nn.Embedding(num_embeddings=self.num_users + self.num_items, embedding_dim=args['embedding_dim'])
        # if args['isPretrain'] == 0:
        #     nn.init.normal_(self.ent_embeddings.weight, std=0.1)
        #     nn.init.normal_(self.rel_embedding, std=0.1)
        # else:
        #     self.ent_embeddings.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
        #     self.rel_embeddings.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
        #     print('use pretarined data')

        self.norm_adj = norm_adj.to(device)

        ### Attention Design
        self.W_R = nn.Parameter(torch.Tensor(self.num_rel, args['ent_embedding_dim'], args['rel_embedding_dim']))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.Neg_Sampler = NegativeSampler(args['data']['name'], kg, ent2id, rel2id, args['kg_neg_size'])
        self.KGEmodel = TransR(args['ent_embedding_dim'], args['rel_embedding_dim'], rel_num=self.num_rel)
        #self.KGEmodel = TransE()
        self.KGEloss = MarginLoss(margin=3.0)
        self.conv_gcn1 = RGCNConv(in_channels=args['embedding_dim'], out_channels=args['hidden_embedding_dim'], num_relations=self.num_rel, num_bases=self.num_rel)
        self.conv_gcn2 = RGCNConv(in_channels=args['hidden_embedding_dim'], out_channels=args['embedding_dim'], num_relations=self.num_rel, num_bases=self.num_rel)
        self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.remap_layer = nn.Linear(args['embedding_dim'] * 3, args['embedding_dim'])
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(args['embedding_dim'] * 2, args['embedding_dim']),
            nn.ReLU(),
            nn.Linear(args['embedding_dim'], args['embedding_dim'])
        )

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.num_ag_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        #### Evaluation
        self.rate_act_fn = nn.Sigmoid()
    
    def forward(self, edge_indexs, edge_types, users, items, neg_items):
        ### KG Embedding Learning(only training the ent_embeddings_kge, and rel_embeddings)       
        merged_edge_index = torch.cat(edge_indexs, dim=1)
        merged_edge_type = torch.cat(edge_types)
        merged_edge_index = merged_edge_index.to(self.device)
        merged_edge_type = merged_edge_type.to(self.device)

        edge_attr = self.compute_attention('kg', merged_edge_index, merged_edge_type, len(merged_edge_type))
        bpr_reg_loss, bpr_loss = self._calc_bpr_loss(merged_edge_index, edge_attr, users, items, neg_items)
        
        kge_reg_loss, kge_loss = self._calc_kge_loss(merged_edge_index, merged_edge_type)

        total_loss = bpr_reg_loss + self.kge_weight * kge_reg_loss

        return total_loss, bpr_loss, kge_loss
    
    def att_score(self, graph_type, edge_index, edge_type, valid_id):
        r_mul_h = torch.matmul(self.ent_embeddings_kge(edge_index[0][valid_id].long()), self.W_r)          
        r_mul_t = torch.matmul(self.ent_embeddings_kge(edge_index[1][valid_id].long()), self.W_r)         
        r_embed = self.rel_embeddings(edge_type[valid_id])                                             
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)  
        return att

    def compute_attention(self, graph_type, edge_index, edge_type, num_edge):
        sub_edge_id = torch.arange(num_edge).to(self.device)
        edge_attr = torch.zeros(len(sub_edge_id)).to(self.device)

        for i in range(self.num_rel):
            mask = (edge_type == i)
            if sum(mask) == 0:
                continue
            tar_edge_id = sub_edge_id[mask]
            self.W_r = self.W_R[i]  # [entity_dim, relation_ dim]
            att = self.att_score(graph_type, edge_index, edge_type, tar_edge_id)
            edge_attr[mask] = att.view(1, -1)

        edge_attr = edge_softmax(edge_index, edge_attr)

        return edge_attr
    
    def _calc_bpr_loss(self, edge_index, edge_attr, users, items, neg_items):
        x = self._forward_aggregator(self.ent_embeddings_kge.weight, edge_index, edge_attr)                
        
        user_embs = x[users]               
        pos_item_embs = x[items]       
        neg_item_embs = x[neg_items]       

        sup_pos_ratings = inner_product(user_embs, pos_item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings

        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))

        bpr_reg_loss = l2_loss(
            user_embs,
            pos_item_embs,
            neg_item_embs
        )
        
        loss = bpr_loss + self.reg_weight * bpr_reg_loss
        return loss, bpr_loss
    
    def _calc_kge_loss(self, edge_index, edge_type):
        edge_index_expand, edge_type_expand = self.Neg_Sampler.Triples_neg_sample(edge_index, edge_type)
        edge_index_expand = edge_index_expand.to(self.device)
        edge_type_expand = edge_type_expand.to(self.device)
    
        head_embs = self.ent_embeddings_kge(edge_index_expand[0].long())
        tail_embs = self.ent_embeddings_kge(edge_index_expand[1].long())
        rels_embs = self.rel_embeddings(edge_type_expand)

        #pos_score, neg_score = self.KGEmodel(head_embs, tail_embs, rels_embs, len(edge_type))
        pos_score, neg_score = self.KGEmodel(head_embs, tail_embs, rels_embs, edge_type_expand, len(edge_type), self.W_R)
        kge_loss = self.KGEloss(pos_score, neg_score)

        kge_reg_loss = l2_loss(
            head_embs,
            tail_embs,
            rels_embs
        )
        loss = kge_loss + self.reg_weight * kge_reg_loss
        return loss, kge_loss

    def _forward_aggregator(self, x, edge_index, edge_attr):
        all_x = []
        for layer in self.aggregator_layers:
            x = layer(x, edge_index, edge_attr)
            norm_x = F.normalize(x, p=2, dim=1)
            all_x.append(norm_x)
        all_x = torch.cat(all_x, dim=-1)
        return all_x

    def _forward_rgcn(self, x, edge_indexs, edge_types):
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
        return x
    
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
        # concat_embs = torch.cat((kge_embeddings, ui_embeddings), dim=-1)
        # fused_embs = self.fusion_mlp(concat_embs)
        # return fused_embs

        ### Bi-directional Attention Fusion
        query = kge_embeddings
        key = ui_embeddings

        scores = torch.bmm(query.unsqueeze(1), key.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (batch_size)
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (batch_size, 1)
        kge_att_ui = attn_weights * kge_embeddings  # (batch_size, emb_dim)  
        ui_att_kge = (1 - attn_weights) * ui_embeddings  # (batch_size, emb_dim)
        
        fused_embs = torch.cat((kge_att_ui, ui_att_kge), dim=-1)  # (batch_size, 2*emb_dim)
        fused_embs = self.fusion_mlp(fused_embs)  # (batch_size, emb_dim)
        
        return fused_embs
        
    def getUsersRating(self, users):
        users_emb = self.ent_embeddings_kge(users)
        items_emb = self.ent_embeddings_kge.weight[self.num_users : self.num_users + self.num_items]
        rating = self.rate_act_fn(torch.matmul(users_emb, items_emb.T))
        return rating
    
    def pretrain_kg_embeddings(self, edge_indexs, edge_types):
        merged_edge_index = torch.cat(edge_indexs, dim=1)
        merged_edge_type = torch.cat(edge_types)
        rels_for_reg = self.rel_embeddings(merged_edge_type.unique().to(self.device))
        edge_index_expand, edge_type_expand = self.Neg_Sampler.Triples_neg_sample(merged_edge_index, merged_edge_type)
        edge_index_expand = edge_index_expand.to(self.device)
        edge_type_expand = edge_type_expand.to(self.device)
    
        head = self.ent_embeddings_kge(edge_index_expand[0].long())
        tail = self.ent_embeddings_kge(edge_index_expand[1].long())
        rels = self.rel_embeddings(edge_type_expand)

        pos_score, neg_score = self.KGEmodel(head, tail, rels, len(merged_edge_type))
        #pos_score, neg_score = self.KGEmodel(head, tail, rels, edge_type_expand, len(merged_edge_type))
        kge_loss = self.KGEloss(pos_score, neg_score)
        reg_loss = l2_loss(
            head,
            tail,
            rels
        )
        loss = kge_loss + self.reg_weight * reg_loss
        return loss, kge_loss