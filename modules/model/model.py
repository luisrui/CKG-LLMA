import torch
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F

from .BasicModel import BasicModel
from ..utils import *
from .KGEmb import *
from .GAT import GAT

## Knowledge-enhanced Large-language-model Contrastive Recommender
class KLMCR(BasicModel):
    def __init__(self, args, rec_data, kg_data):
        super(KLMCR, self).__init__()

        self.args = args
        self.device = args['device']
        
        self.dataset_name = args["data"]['name']
        #self.kg_dataset = kg_data
        self.__init_weight(rec_data, kg_data)
        
        ### BPR
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.reg_weight = args['loss_reg_weight']
        ### KGE
        #self.KGEmodel = TransE()
        #self.KGEloss = MarginLoss(margin=3.0)
        self.kgcn = args['kgcn']
        self.gat = GAT(self.latent_dim, self.latent_dim,
                       self.num_relations + 1, dropout=0.4, alpha=0.2).train()
    
    def __init_weight(self, rec_data, kg_data):
        self.num_users = self.args['num_users']
        self.num_items = self.args['num_items']
        self.ent2id = self.args['ent2id']
        self.rel2id = self.args['rel2id']
        self.num_entities = len(self.ent2id)
        self.num_relations = len(self.rel2id)
        self.num_attrs = self.num_entities - self.num_items - self.num_users

        self.latent_dim = self.args['embedding_dim']
        self.n_layers = self.args['lightGCN_n_layers']
        self.keep_prob = self.args['keep_prob']
        self.A_split = self.args['A_split']
        self.entity_num_per_item = self.args['entity_num_per_item']
        self.item_num_per_entity = self.args['item_num_per_entity']

        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities+1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations+1, embedding_dim=self.latent_dim)

        if self.args['LoadPretrain'] == 0:
            # nn.init.normal_(self.embedding_user.weight, std=0.1)
            # nn.init.normal_(self.embedding_item.weight, std=0.1)
            nn.init.normal_(self.embedding_entity.weight, std=0.1)
            nn.init.normal_(self.embedding_relation.weight, std=0.1)
        else:
            self.embedding_entity.weight.data.copy_(
                torch.from_numpy(self.args['entity_emb']))
            self.embedding_relation.weight.data.copy_(
                torch.from_numpy(self.args['relation_emb']))
            print('use pretrained data')
        self.f = nn.Sigmoid()
        self.Graph = rec_data.get_norm_adj.to(self.device)
        # self.ItemNet = self.kg_dataset.get_item_net_from_kg(self.num_items)
        self.kg_dict, self.item2relations = kg_data.get_kg_dict()
        self.i2e, self.i2r = self.kg_dict.copy(), self.item2relations.copy()
        for key in self.i2e:
            self.i2e[key] = torch.IntTensor(self.i2e[key])
            self.i2r[key] = torch.IntTensor(self.i2r[key])
        self.item_ids = torch.IntTensor(list(self.kg_dict.keys())).to(self.device)
        self.attr_ids = torch.IntTensor(list(range(self.num_users + self.num_items, self.num_entities))).to(self.device)
        self.item_entities = torch.stack(list(self.i2e.values())).to(self.device)
        self.item_relations = torch.stack(list(self.i2r.values())).to(self.device)
        self.entity_items = self.get_reverse_kg(self.item_entities)
        #self.W_Q = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)            # (kg_batch_size, relation_dim)
        # (kg_batch_size, entity_dim)
        h_embed = self.embedding_entity(h)
        pos_t_embed = self.embedding_entity(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(neg_t)      # (kg_batch_size, entity_dim)

        pos_score = torch.sum(
            torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)     # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        kge_reg_loss = self.reg_loss(
            h_embed,
            r_embed,
            pos_t_embed,
            neg_t_embed,
            require_pow=True
        )
        loss = kg_loss + self.reg_weight * kge_reg_loss
        return loss
    
    def calc_kg_loss_transR(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)            # (kg_batch_size, relation_dim)
        W_r = self.gat.layer.W_h[r]                           # (kg_batch_size, entity_dim, entity_dim)
        
        h_embed = self.embedding_entity(h)
        pos_t_embed = self.embedding_entity(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(neg_t)      # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)

        pos_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        kge_reg_loss = self.reg_loss(
            h_embed,
            r_embed,
            pos_t_embed,
            neg_t_embed,
            require_pow=True
        )
        loss = kg_loss + self.reg_weight * kge_reg_loss
        # loss = kg_loss
        return loss

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_entity.weight[:self.num_users]
        if self.args['isContrastive']:
            items_emb = self.cal_item_embedding_from_kg(self.item_entities, self.item_relations, self.item_ids, self.entity_items)
        else:
            items_emb = self.embedding_entity.weight[self.num_users:self.num_users+self.num_items]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.args['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        return light_out

    def getUsersRating(self, users):
        all_embs = self.computer()
        users_emb = all_embs[users.long()]
        items_emb = all_embs[self.num_users:]
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_embs = self.computer()
        users_emb = all_embs[users]
        pos_emb = all_embs[pos_items]
        neg_emb = all_embs[neg_items]
        # users_emb_ego = self.embedding_entity(users)
        # pos_emb_ego = self.embedding_entity(pos_items)
        # neg_emb_ego = self.embedding_entity(neg_items)
        return users_emb, pos_emb, neg_emb
    
    def calc_bpr_loss(self, users, pos_items, neg_items):
        (users_emb, posi_emb, negi_emb) = \
        self.getEmbedding(users.long(), pos_items.long(), neg_items.long())

        pos_scores = inner_product(users_emb, posi_emb)
        neg_scores = inner_product(users_emb, negi_emb)

        bpr_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(
            users_emb,
            posi_emb,
            negi_emb,
            require_pow=True
        )
        loss = bpr_loss + self.reg_weight * reg_loss
        return loss, bpr_loss
    
    def view_computer_all(self, g_droped, item_entities, item_relations):
        """
        propagate methods for contrastive lightGCN
        """
        users_emb = self.embedding_entity.weight[:self.num_users]
        items_emb = self.cal_item_embedding_from_kg(item_entities, item_relations, self.item_ids)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        user_embs, item_embs = torch.split(light_out, [self.num_users, self.num_items])
        return user_embs, item_embs

    def cal_item_embedding_from_kg(self, item2ent, item2rel, i_ids, ent2item = None):
        if ent2item is None:
            ent2item = self.get_reverse_kg(item2ent)
        e_ids = self.attr_ids
        if (self.kgcn == "GAT"):
            return self.cal_item_embedding_gat(item2ent)
        elif self.kgcn == "RGAT":
            return self.cal_item_embedding_rgat(item2ent, item2rel, i_ids)
        elif (self.kgcn == "MEAN"):
            return self.cal_item_embedding_mean(item2ent)
        elif (self.kgcn == "Ours"):
            return self.cal_item_embedding_KLMCR(item2ent, item2rel)
        elif (self.kgcn == "OursSingle"):
            return self.cal_item_embedding_SingleVariant(item2ent, item2rel, i_ids, ent2item, e_ids)
        elif (self.kgcn == "NO"):
            return self.embedding_entity.weight[self.num_users:self.num_users+self.num_items]

    def cal_item_embedding_KLMCR(self, kg: dict, item2rel:dict = None):
        if item2rel is None:
            item2rel = self.item_relations
        item_embs = self.embedding_entity(torch.IntTensor(
            list(kg.keys())).to(self.device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        item_relations = torch.stack(list(item2rel.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        relation_embs = self.embedding_relation(
            item_relations)  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation_specific(item_embs, entity_embs, relation_embs, item_relations, padding_mask)
    
    def cal_item_embedding_SingleVariant(self, item2ent, item2rel, i_ids, ent2item, e_ids):
        item_embs = self.embedding_entity(i_ids)  # item_num, emb_dim
        attr_embs = self.embedding_entity(e_ids) # entity_num, emb_dim
        i2e_embs = self.embedding_entity(item2ent) # item_num, e_num, emb_dim
        e2i_embs = self.embedding_entity(ent2item) # entity_num, i_num, emb_dim
        padding_mask_i2e = torch.where(item2ent != self.num_entities, torch.ones_like(
            item2ent), torch.zeros_like(item2ent)).float() # N_item, e_num    
        padding_mask_e2i = torch.where(ent2item != self.num_entities, torch.ones_like(
            ent2item), torch.zeros_like(ent2item)).float() # N_ent, i_num 
        item_embs = self.gat.fusion_item_embs(item_embs, i2e_embs, padding_mask_i2e) # N_item, dim
        We = self.gat.fusion_attr_embs(attr_embs, e2i_embs, padding_mask_e2i) # N_ent, dim
        i2e_embs = self.renew_entity_embs(item2ent, We) # N_item, e_num, dim
        relation_embs = self.embedding_relation(
            item2rel)  # item_num, entity_num_each, emb_dim
        return self.gat.forward_relation(item_embs, i2e_embs, relation_embs, padding_mask_i2e)
     
    def cal_item_embedding_rgat(self, item_entities, item_relations, ids):
        item_embs = self.embedding_entity(ids)  # item_num, emb_dim
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        relation_embs = self.embedding_relation(
            item_relations)  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def cal_item_embedding_mean(self, kg: dict):
        item_embs = self.embedding_entity(torch.IntTensor(
            list(kg.keys())).to(self.device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        # padding为0
        entity_embs = entity_embs * \
            padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # item_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / \
            padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # item_num, emb_dim
        return item_embs+entity_embs_mean
    
    @torch.no_grad()
    def get_reverse_kg(self, item_entities):
        # padding = self.num_entities
        # entity_indices = item_entities.view(-1)
        # item_indices = self.item_ids.repeat_interleave(self.entity_num_per_item)
        
        # valid_mask = (entity_indices != padding)
        # entity_indices = entity_indices[valid_mask] - self.num_users - self.num_items
        # item_indices = item_indices[valid_mask]
        # attr_items = torch.full((self.num_attrs, self.item_num_per_entity), padding, dtype=torch.long, device=self.device)
        # counts = torch.zeros(self.num_attrs, dtype=torch.long, device=self.device)

        # for i in range(self.item_num_per_entity):
        #     mask = (counts < self.item_num_per_entity) & (entity_indices < self.num_attrs)
        #     counts = counts + mask.long()
        #     scatter_indices = (entity_indices + mask.long() * self.num_attrs).clamp(0, self.num_attrs - 1)
        #     attr_items.scatter_(1, counts.view(-1, 1).expand(-1, scatter_indices.shape[0]), item_indices)
        # attr_items = [[] for _ in range(self.num_attrs)] # N_ent, e_num
        # for ent_idx, itm_idx in zip(entity_indices.cpu(), item_indices.cpu()):
        #     ent_idx = ent_idx - self.num_users - self.num_items
        #     attr_items[ent_idx].append(itm_idx)
        # for i in range(self.num_attrs):
        #     if len(attr_items[i]) > self.item_num_per_entity:
        #         attr_items[i] = attr_items[i][:self.item_num_per_entity]
        #     attr_items[i] = torch.tensor(attr_items[i], dtype=torch.long, device=self.device)
        #     if len(attr_items[i]) < self.item_num_per_entity:
        #         attr_items[i] = F.pad(attr_items[i], (0, self.item_num_per_entity - attr_items[i].shape[0]), 'constant', padding)
        # attr_items = torch.stack(attr_items)    
        # attr_items = attr_items.masked_fill(attr_items == padding, padding)   
        # return attr_items
        return 0

    def renew_entity_embs(self, item_entities, We):
        padding = self.num_entities
        mask = item_entities != padding
        result = torch.zeros(*item_entities.shape, We.shape[-1], device=self.device) # N_item, e_num, dim
        valid_indices = item_entities[mask] - self.num_users - self.num_items
        result[mask] = We[valid_indices]
        result[~mask] = self.embedding_entity(torch.full_like(item_entities, padding)[~mask])
        return result
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

