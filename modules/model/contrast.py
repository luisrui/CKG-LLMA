from cppimport import imp
from numpy import negative, positive
from torch_sparse.tensor import to
from random import random, sample
from .model import KLMCR
from .BasicModel import CrossAttentionLayer
from ..utils.utils import edge_softmax
import torch
import torch.nn as nn
import math
#from torch_geometric.utils import degree, to_undirected
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
"""
graph shape:
[    0,     0,     0,  ..., 69714, 69715, 69715],
[    0, 31668, 31669,  ..., 69714, 31666, 69715]

values=tensor([0.0526, 0.0096, 0.0662,  ..., 0.5000, 0.1443, 0.5000])
"""

class Contrast(nn.Module):
    def __init__(self, args, model, rec_data):
        super(Contrast, self).__init__()
        self.model : KLMCR = model
        self.device = args['device']
        self.isSeperated = args['ContrastiveSeperate']
        self.isFused = args['ContrastiveFused']

        self.tau = args['nce_temperatue']
        self.kg_p_drop = args['kg_p_drop']
        self.ui_p_drop = args['ui_p_drop']
        self.mix_ratio = args['mix_ratio']
        self.num_users = self.model.num_users
        self.num_items = self.model.num_items
        self.item_np = rec_data.item_np
        self.user_np = rec_data.user_np
        self.confi_thres = args['confidence_threshold']
        self.emb_size = self.model.latent_dim
        self.gb_tau = args['gumbel_tau']
        self.cross_attention = CrossAttentionLayer(self.emb_size, num_heads=8)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def info_nce_loss_overall(self, z1, z2, z_all):
        def f(x): return torch.exp(x / self.tau)
        # batch_size
        between_sim = f(self.sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(self.sim(z1, z_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss

    def get_kg_views(self, rectify_info : dict):
        h2t = self.model.kg_dict
        h2r = self.model.item2relations
        if self.isSeperated:
            view1, rel_v1 = self.drop_edge_random(
                h2t, h2r, self.kg_p_drop, self.model.num_entities)
            view2, rel_v2 = self.drop_edge_random(
                h2t, h2r, self.kg_p_drop, self.model.num_entities)
            #return view1, rel_v1, view2, rel_v2
            view_ii, rel_ii = self.drop_LLM_rectify(
                h2t, h2r, rectify_info, 'ii', self.model.num_entities, self.model.num_relations)
            view_ui, rel_ui = self.drop_LLM_rectify(
                h2t, h2r, rectify_info, 'ui', self.model.num_entities, self.model.num_relations)
            return  {
                'view_ii': view_ii, 'rel_ii': rel_ii,
                'view_ui': view_ui, 'rel_ui': rel_ui,
                'view1' : view1, 'relv1' : rel_v1,
                'view2' : view2, 'relv2' : rel_v2
            }
        elif self.isFused:
            i_ids, i_ents_v1, i_rels_v1 = self.drop_edge_random_rectify(
                h2t, h2r, rectify_info, 'ii', self.kg_p_drop, self.model.num_entities, self.model.num_relations)
            i_ids, i_ents_v2, i_rels_v2 = self.drop_edge_random_rectify(
                h2t, h2r, rectify_info, 'ui', self.kg_p_drop, self.model.num_entities, self.model.num_relations)
            return {
                'item_ids' : i_ids,
                'entv1' : i_ents_v1, 'relv1' : i_rels_v1,
                'entv2' : i_ents_v2, 'relv2' : i_rels_v2
            }
        else:
            view1, rel_v1 = self.drop_edge_random(
                h2t, h2r, self.kg_p_drop, self.model.num_entities)
            view2, rel_v2 = self.drop_edge_random(
                h2t, h2r, self.kg_p_drop, self.model.num_entities)
            return {
                'view1' : view1, 'relv1' : rel_v1,
                'view2' : view2, 'relv2' : rel_v2
            }

    def get_ui_views_weighted(self, item_stabilities, del_cands = None):
        # graph = self.model.Graph
        # n_users = self.num_users

        # # generate mask
        # item_degrees = degree(graph.indices()[0])[n_users:].tolist()
        # deg_col = torch.FloatTensor(item_degrees).to(self.device)
        # s_col = torch.log(deg_col)
        # # degree normalization
        # # deg probability of keep
        # degree_weights = (s_col - s_col.min()) / (s_col.max() - s_col.min())
        # degree_weights = degree_weights.where(
        #     degree_weights > 0.3, torch.ones_like(degree_weights) * 0.3)  # p_tau

        # kg probability of keep
        item_stabilities = torch.exp(item_stabilities)
        kg_weights = (item_stabilities - item_stabilities.min()) / \
            (item_stabilities.max() - item_stabilities.min())
        kg_weights = kg_weights.where(
            kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)

        # overall probability of keep
        weights = (1-self.ui_p_drop)/torch.mean(input=kg_weights)*(kg_weights)
        weights = weights.where(
            weights < 0.95, torch.ones_like(weights) * 0.95)

        item_mask = torch.bernoulli(weights).to(torch.bool)
        #print(f"keep ratio: {item_mask.sum()/item_mask.size()[0]:.2f}")
        # drop
        g_weighted = self.ui_drop_weighted(item_mask, del_cands)
        g_weighted.requires_grad = False
        return g_weighted

    def item_kg_stability(self, view1, relv1, view2, relv2, ids):
        kgv1_ro = self.model.cal_item_embedding_from_kg(view1, relv1, ids)
        kgv2_ro = self.model.cal_item_embedding_from_kg(view2, relv2, ids)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return sim
    
    def ui_drop_weighted(self, item_mask, del_cands = None):
        # item_mask: [item_num]
        item_mask = item_mask.tolist()
        n_nodes = self.num_users + self.num_items
        # [interaction_num]
        item_np = self.item_np
        keep_idx = list()
            # overall sample rate = 0.4*0.9 = 0.36
        for i, j in enumerate(item_np.tolist()):
            if del_cands and j not in del_cands:
                continue
            if item_mask[j - self.num_users] and random() > 0.6:
                keep_idx.append(i)
        # for i, j in enumerate(item_np.tolist()):
        #     if item_mask[j - self.num_users] and random() > 0.6:
        #         keep_idx.append(i)
        # add random samples
        interaction_random_sample = sample(
            list(range(len(item_np))), int(len(item_np)*self.mix_ratio))
        keep_idx = list(set(keep_idx+interaction_random_sample))
        # for i, j in enumerate(item_np.tolist()):
        #     if item_mask[j]:
        #         keep_idx.append(i)

        print(f"finally keep ratio: {len(keep_idx)/len(item_np.tolist()):.2f}")
        keep_idx = np.array(keep_idx)
        user_np = self.user_np[keep_idx]
        item_np = item_np[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse_coo_tensor(index, data, torch.Size(
            coo.shape)).coalesce().to(self.device)
        g.requires_grad = False
        return g

    def transform_origin_graph(self, graph):
        # to coo
        graph_vers = graph.cpu().coalesce()
        indices = graph_vers.indices().numpy()
        values = graph_vers.values().numpy()
        shape = graph_vers.shape
        graph_vers = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)

        coo = graph_vers.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse_coo_tensor(index, data, torch.Size(
            coo.shape)).coalesce().to(self.device)
        g.requires_grad = False
        return g

    def get_views(self, rectify_info):
        # drop (epoch based)
        # kg drop -> 2 views -> view similarity for item
        collect_info = self.get_kg_views(rectify_info)
        if self.isSeperated:
            v_ii, rel_ii, v_ui, rel_ui = collect_info['view_ii'], collect_info['rel_ii'], collect_info['view_ui'], collect_info['rel_ui']
            v_1, r_1, v_2, r_2 = collect_info['view1'], collect_info['relv1'], collect_info['view2'], collect_info['relv2']
        elif self.isFused:
            item_ids = collect_info['item_ids']
            v_ii, rel_ii, v_ui, rel_ui = collect_info['entv1'], collect_info['relv1'], collect_info['entv2'], collect_info['relv2']
        else: 
            v_1, r_1, v_2, r_2 = collect_info['view1'], collect_info['relv1'], collect_info['view2'], collect_info['relv2']

        if self.isSeperated:
            stability = self.item_kg_stability(v_1, r_1, v_2, r_2).to(self.device)
            uiv1 = self.get_ui_views_weighted(stability)
            uiv2 = self.get_ui_views_weighted(stability)
            stability_adjusted = self.item_kg_stability(v_ii, rel_ii, v_ui, rel_ui).to(self.device)
            uiv3 = self.get_ui_views_weighted(stability_adjusted)
            uiv4 = self.get_ui_views_weighted(stability_adjusted, rectify_info['del_cands'])
        #uiv1, uiv2 = self.transform_origin_graph(self.model.Graph), self.transform_origin_graph(self.model.Graph)
            contrast_views = {
                "uiv1": uiv1, 'relv1': r_1, 'v_1' : v_1,
                "uiv2": uiv2, 'relv2': r_2, 'v_2' : v_2,
                'uiv3': uiv3, 'rel_ii': rel_ii, 'v_ii': v_ii,
                'uiv4': uiv4, 'rel_ui': rel_ui, 'v_ui': v_ui,
            }
        elif self.isFused:
            stability = self.item_kg_stability(v_ii, rel_ii, v_ui, rel_ui, item_ids).to(self.device)
            uiv1 = self.get_ui_views_weighted(stability)
            uiv2 = self.get_ui_views_weighted(stability)
            contrast_views = {
                "uiv1": uiv1, 'relv1': rel_ii, 'v_1' : v_ii,
                "uiv2": uiv2, 'relv2': rel_ui, 'v_2' : v_ui,
            }
        else:
            stability = self.item_kg_stability(v_1, r_1, v_2, r_2).to(self.device)
            uiv1 = self.get_ui_views_weighted(stability)
            uiv2 = self.get_ui_views_weighted(stability)
            contrast_views = {
                "uiv1": uiv1, 'relv1': r_1, 'v_1' : v_1,
                "uiv2": uiv2, 'relv2': r_2, 'v_2' : v_2,
            }
        return contrast_views
    
    def confidence_drop(self, item_embs, entity_embs, rel_embs, item_relations, padding_mask):
        '''
        item_embs: item_num, ent_num, dim
        entity_embs: item_num, ent_num, dim
        rel_embs: item_num, ent_num, dim
        adj: item_num, ent_num
        item_relations (item_num, ent_num)
        attention (item_num, ent_num)
        '''
        W_h = self.model.gat.layer.W_h # num_rels, e_embs, e_embs
        w_hr  = W_h[item_relations] # (item_num, ent_num, dim, dim)

        head_query = torch.einsum('ijkl,ijl->ijk', w_hr, item_embs)
        tail_key = torch.einsum('ijkl,ijl->ijk', w_hr, entity_embs)

        #edge_attn = (query * (key * value)).sum(dim=-1) / math.sqrt(emb_size)
        #edge_attn = torch.multiply(head_query + tail_key, rel_embs).sum(dim=-1) / math.sqrt(emb_size)
        #edge_weight = torch.multiply(head_query + tail_key, rel_embs).sum(dim=-1)
        #zero_vec = -9e15*torch.ones_like(edge_weight)
        #edge_weight = torch.where(padding_mask > 0, edge_weight, zero_vec) # item_num, ent_num

        'please input cross attention logic here:'
        rel_value = rel_embs
        cross_attn_weights = self.cross_attention(head_query, tail_key, rel_value, padding_mask)
        
        # Combine self-attention and cross-attention
        cross_attention_scores = cross_attn_weights.mean(dim=-1)  # Average over heads
        #cross_attention_scores = (edge_weight + cross_attention_scores) / 2

        zero_vec = -9e15*torch.ones_like(cross_attention_scores)
        cross_attention_scores = torch.where(padding_mask > 0, cross_attention_scores, zero_vec) # item_num, ent_num
        combined_attention = F.sigmoid(cross_attention_scores)

        # Standard Normalization
        # edge_confi = torch.exp(edge_attn)
        # edge_confi = (edge_confi - edge_confi.min()) / (edge_confi.max() - edge_confi.min())
        # edge_confi = (1 - self.kg_p_drop) / torch.mean(edge_confi) * (edge_confi)

        # normalization by head_node degree
        # norm = scatter_sum(torch.ones_like(batch_heads), batch_heads, dim=0, dim_size=entity_emb.weight.shape[0])
        # norm = torch.index_select(norm, 0, batch_heads)
        # edge_attn_score = edge_attn_score * norm

        return combined_attention
    
    def filter_LLM_info(self, item_ids, item_entities, item_relations):
        item_embs = self.model.embedding_entity(item_ids)
        entity_embs = self.model.embedding_entity(item_entities) 
        item_embs = item_embs.unsqueeze(1).expand(entity_embs.size())
        relation_embs = self.model.embedding_relation(item_relations)  
        padding_mask = torch.where(item_entities != self.model.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()

        edge_confi = self.confidence_drop(item_embs, entity_embs, relation_embs, item_relations, padding_mask)

        ### The Gumbel softmax filtering trick:
        logits = torch.stack([edge_confi, 1 - edge_confi], dim=-1)
        gumbel_out = F.gumbel_softmax(logits, tau=self.gb_tau, hard=True)
        decisions = gumbel_out[:, :, 0] # The decisions to keep the triples
        print(f'kg keep ratio : {torch.sum(torch.logical_and(decisions, padding_mask))/torch.sum(padding_mask)}')
        return decisions
    
    def _delete_triples(self, res_e, tri2del, padding_e):
        for head, _, tail in tri2del:
            if head in res_e.keys():
                mask = res_e[head] == tail
                res_e[head][mask] = padding_e

    def _add_triples(self, res_e, res_r, tri2add, padding_e):
        for head, rel, tail in tri2add:
            mask = res_e[head] == padding_e
            if mask.any():
                idx = mask.nonzero()[0][0]
                res_e[head][idx] = tail
                res_r[head][idx] = rel

    def drop_edge_random_rectify(self, head2tail, head2rel, rectify_info, subgraph:str, p_drop, padding_e, padding_r):
        res_e = head2tail.copy()
        res_r = head2rel.copy()
        tri2add = rectify_info[f'{subgraph}_add']
        tri2del = rectify_info[f'{subgraph}_del']
        self._delete_triples(res_e, tri2del, padding_e)
        self._add_triples(res_e, res_r, tri2add, padding_e)
        for key in res_e:
            res_e[key] = torch.IntTensor(res_e[key]).to(self.device)
            res_r[key] = torch.IntTensor(res_r[key]).to(self.device)
        item_ids = torch.IntTensor(list(res_e.keys())).to(self.device)
        item_entities = torch.stack(list(res_e.values()))
        item_relations = torch.stack(list(res_r.values()))
        ## rectify based on LLM
        entity_mask = self.filter_LLM_info(item_ids, item_entities, item_relations)
        replacement = torch.full_like(item_entities, self.model.num_entities)
        item_entities = torch.where(entity_mask.bool(), item_entities, replacement)
        ## Random Dropout
        #item_entities = item_entities * torch.bernoulli(1 - p_drop)
        return item_ids, item_entities, item_relations
    
    def drop_edge_random(self, head2tail, head2rel, p_drop, padding):
        res_e = dict()
        res_r = head2rel.copy()
        item_ids = torch.IntTensor(list(res_e.keys())).to(self.device)
        item_entities = torch.stack(list(res_e.values()))
        item_relations = torch.stack(list(res_r.values()))

        random_tensor = torch.rand_like(item_entities, dtype=torch.float)
        mask = random_tensor < p_drop
        padding_tensor = torch.full_like(item_entities, padding)
        item_entities = torch.where(mask, padding_tensor, item_entities).to(self.device)
        item_relations = item_relations.to(self.device)
        return item_ids, item_entities, item_relations
    
    def drop_LLM_rectify(self, head2tail, head2rel, rectify_info, subgraph:str, padding_e, padding_r):
        tri2add = torch.IntTensor(rectify_info[f'{subgraph}_add'])
        tri2del = torch.IntTensor(rectify_info[f'{subgraph}_del'])
        res_t, res_r = head2tail.copy(), head2rel.copy()
        # delete triples based on LLM
        combined_triples = torch.cat([tri2add, tri2del], dim=0)
        num_add, num_del = tri2add.shape[0], tri2del.shape[0]
        heads, relations, tails = combined_triples[:, 0].to(self.device), combined_triples[:, 1].to(self.device), combined_triples[:, 2].to(self.device)
        batch_confi = self.confidence_drop(heads, relations, tails)
        
        #triple_mask = batch_confi > self.confi_thres
        ### The Gumbel softmax filtering trick:
        logits = torch.stack([batch_confi, 1 - batch_confi], dim=-1)
        gumbel_out = F.gumbel_softmax(logits, tau=self.gb_tau, hard=True)
        decisions = gumbel_out[:, 0]
        add_decisions = decisions[:num_add]
        del_decisions = decisions[num_add:]
        filtered_add = tri2add[add_decisions == 1] # Filter add triples (keep if decision is 1)
        filtered_del = tri2del[del_decisions == 0] # Filter del triples (keep if decision is 0, which means delete)
        # filtered_del = tri2del.numpy().tolist()
        # filtered_add = tri2add.numpy().tolist()
        for triple in filtered_del:
            head, _, tail = triple
            if head not in res_t or tail not in res_t[head]:
                continue
            idx = res_t[head].index(tail)
            res_t[head][idx] = padding_e
        ## add triples based on LLM
        for head, rel, tail in filtered_add:
            if head not in res_t and tail < self.num_users + self.num_items: ## In case it is triples of other kinds, not i-a
                continue                    
            if padding_e in res_t[head]:
                idx = res_t[head].index(padding_e)
                res_t[head][idx] = tail
                res_r[head][idx] = rel 
        for key in res_t:
            res_t[key] = torch.IntTensor(res_t[key]).to(self.device)
            res_r[key] = torch.IntTensor(res_r[key]).to(self.device)
        return res_t, res_r
    
    # def confidence_drop(self, item_embs, entity_embs, rel_embs, i2r, padding_mask):
    #     entity_emb = self.model.embedding_entity
    #     relation_emb = self.model.embedding_relation
    #     emb_size = self.model.latent_dim
    #     W_h = self.model.gat.layer.W_h # num_rels, e_embs, e_embs
    #     #W_e = self.model.gat.layer.W_e # num_rels, e_embs, e_embs
    #     w_Q  = W_h[batch_relations] # batch_size, e_embs, e_embs
    #     #w_Qh = W_h[batch_relations] # batch_size, e_embs, e_embs
    #     #w_Qe  = W_e[batch_relations] # batch_size, e_embs, e_embs
        
    #     query = torch.bmm(entity_emb(batch_heads).unsqueeze(1), w_Q).squeeze(1)
    #     key = torch.bmm(entity_emb(batch_tails).unsqueeze(1), w_Q).squeeze(1)
    #     value = relation_emb(batch_relations)

    #     #edge_attn = (query * (key * value)).sum(dim=-1) / math.sqrt(emb_size)
    #     edge_attn = torch.multiply(value, query + key).sum(dim=-1) / math.sqrt(emb_size)

    #     # softmax by head_node
    #     #edge_attn_logits = edge_attn.mean(-1).detach()
    #     # edge_attn_score = edge_softmax(batch_heads, batch_tails, edge_attn)
    #     # edge_attn_score = scatter_softmax(edge_attn_logits, batch_heads)

    #     edge_confi = torch.exp(edge_attn)
    #     edge_confi = (edge_confi - edge_confi.min()) / (edge_confi.max() - edge_confi.min())
    #     edge_confi = (1 - self.kg_p_drop) / torch.mean(edge_confi) * (edge_confi)
    #     edge_attn_score = edge_confi

    #     # normalization by head_node degree
    #     # norm = scatter_sum(torch.ones_like(batch_heads), batch_heads, dim=0, dim_size=entity_emb.weight.shape[0])
    #     # norm = torch.index_select(norm, 0, batch_heads)
    #     # edge_attn_score = edge_attn_score * norm

    #     return edge_attn_score