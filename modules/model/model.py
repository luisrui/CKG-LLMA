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
                       dropout=0.4, alpha=0.2).train()
    
    def __init_weight(self, rec_data, kg_data):
        self.num_users = self.args['num_users']
        self.num_items = self.args['num_items']
        self.ent2id = self.args['ent2id']
        self.rel2id = self.args['rel2id']
        self.num_entities = len(self.ent2id)
        self.num_relations = len(self.rel2id)

        self.latent_dim = self.args['embedding_dim']
        self.n_layers = self.args['lightGCN_n_layers']
        self.keep_prob = self.args['keep_prob']
        self.A_split = self.args['A_split']

        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities+1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations+1, embedding_dim=self.latent_dim)
        # relation weights
        # self.W_R = nn.Parameter(torch.Tensor(
        #     self.num_relations, self.latent_dim, self.latent_dim))
        # nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

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
        self.i2r_cal, self.i2t_cal = self.item2relations.copy(), self.kg_dict.copy()
        for item in self.i2r_cal:
            self.i2r_cal[item] = torch.IntTensor(self.i2r_cal[item]).to(self.device)
            self.i2t_cal[item] = torch.IntTensor(self.i2t_cal[item]).to(self.device)
        #self.kg_dict = kg_data.kg_dict

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
        # loss = kg_loss
        return loss
    
    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_entity.weight[:self.num_users]
        items_emb = self.cal_item_embedding_from_kg(self.i2t_cal, self.i2r_cal)
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
    
    def view_computer_all(self, g_droped, kg_droped, rel_droped=None):
        """
        propagate methods for contrastive lightGCN
        """
        if rel_droped is None:
            rel_droped = self.i2r_cal
        users_emb = self.embedding_entity.weight[:self.num_users]
        items_emb = self.cal_item_embedding_from_kg(kg_droped, rel_droped)
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

    def cal_item_embedding_from_kg(self, head2tail: dict, head2rel:dict = None):
        if head2tail is None:
            head2tail = self.kg_dict

        if (self.kgcn == "GAT"):
            return self.cal_item_embedding_gat(head2tail)
        elif self.kgcn == "RGAT":
            return self.cal_item_embedding_rgat(head2tail, head2rel)
        elif (self.kgcn == "MEAN"):
            return self.cal_item_embedding_mean(head2tail)
        elif (self.kgcn == "NO"):
            return self.embedding_entity.weight[self.num_users:self.num_users+self.num_items]
        
    def cal_item_embedding_rgat(self, kg: dict, head2rel:dict = None):
        if head2rel is None:
            head2rel = self.i2r_cal
        item_embs = self.embedding_entity(torch.IntTensor(
            list(kg.keys())).to(self.device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        item_relations = torch.stack(list(head2rel.values()))
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

