import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import ones_


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, num_rels, dropout, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.layer = GraphAttentionLayer(nfeat, nhid, num_rels, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward_relation(self, item_embs, entity_embs, w_r, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation(x, y, w_r, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward_relation_specific(self, item_embs, entity_embs, w_r, r_ids, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation_specific(x, y, w_r, r_ids, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward_relation_general(self, item_embs, entity_embs, w_r, r_ids, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation_general(x, y, w_r, r_ids, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x



class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_rels, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.fc = nn.Linear(2*out_features, out_features)

        self.W_h = nn.Parameter(torch.Tensor(num_rels, in_features, out_features))
        nn.init.xavier_uniform_(self.W_h.data, gain=1.414)
        self.W_e = nn.Parameter(torch.Tensor(num_rels, in_features, out_features))
        nn.init.xavier_uniform_(self.W_e.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        
        # N, e_num, dim
        Wh = item_embs.unsqueeze(1).expand(entity_embs.size())
        # N, e_num, dim
        We = entity_embs
        a_input = torch.cat((Wh,We),dim=-1) # (N, e_num, 2*dim)
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(self.fc(a_input), relations).sum(-1) # N,e
        e = self.leakyrelu(e_input) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        We = torch.matmul(entity_embs, self.W) # entity_embs: (N, e_num, in_features), We.shape: (N, e_num, out_features)
        a_input = self._prepare_cat(Wh, We) # (N, e_num, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def forward_relation_specific(self, item_embs, entity_embs, relations, relations_id, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        # relations_id (N, e_num)
        
        #Wr = self.W[relations_id]  # Wr (N, e_num, dim, dim)

        Wr_h = self.W_h[relations_id]  # (N, e_num, dim, dim)
        Wr_e = self.W_e[relations_id]  # (N, e_num, dim, dim)

        # N, e_num, dim
        E_h = item_embs.unsqueeze(1).expand(entity_embs.size())
        # N, e_num, dim
        E_ent = entity_embs
        #a_input = torch.cat((Wh,We),dim=-1) # (N, e_num, 2*dim)

        #transformed = torch.einsum('ijkl,ijk->ijl', Wr, a_input) # (N, e_num, dim)
        transformed_h = torch.einsum('ijkl,ijl->ijk', Wr_h, E_h)  # (N, e_num, dim)
        transformed_e = torch.einsum('ijkl,ijl->ijk', Wr_e, E_ent)  # (N, e_num, dim)
        transformed = transformed_h + transformed_e  # (N, e_num, dim)
       
        e_input = torch.multiply(transformed, relations).sum(dim=-1) # (N, e_num)
        e = self.leakyrelu(e_input) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward_relation_general(self, item_embs, entity_embs, relations, relations_id, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        # relations_id (N, e_num)
        
        Wr_h = self.W_h[relations_id]  # (N, e_num, dim, dim)

        E_h = item_embs.unsqueeze(1).expand(entity_embs.size())# N, e_num, dim
        E_ent = entity_embs # N, e_num, dim

        #transformed = torch.einsum('ijkl,ijk->ijl', Wr, a_input) # (N, e_num, dim)
        transformed_h = torch.einsum('ijkl,ijl->ijk', Wr_h, E_h)  # (N, e_num, dim)
        transformed_e = torch.einsum('ijkl,ijl->ijk', Wr_h, E_ent)  # (N, e_num, dim)
        transformed = transformed_h + transformed_e  # (N, e_num, dim)
        multiplied = torch.multiply(transformed, relations)
        # N,e,2dim -> N,e,dim
        #e_input = torch.multiply(self.fc(a_input), relations).sum(-1) # N,e_num
        e_input = multiplied.sum(dim=-1)  # (N, e_num)
        e = self.leakyrelu(e_input) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size()) # (N, e_num, out_features)
        return torch.cat((Wh, We), dim=-1) # (N, e_num, 2*out_features)