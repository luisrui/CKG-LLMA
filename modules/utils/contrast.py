from cppimport import imp
from numpy import negative, positive
from torch_sparse.tensor import to
from random import random, sample
from ..model import KLMCR
import torch
import torch.nn as nn
from torch_geometric.utils import degree, to_undirected
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
"""
graph shape:
[    0,     0,     0,  ..., 69714, 69715, 69715],
[    0, 31668, 31669,  ..., 69714, 31666, 69715]

values=tensor([0.0526, 0.0096, 0.0662,  ..., 0.5000, 0.1443, 0.5000])
"""


def drop_edge_random(edge_index, p):
    drop_mask = torch.empty((edge_index.size(
        1),), dtype=torch.float32, device=edge_index.device).uniform_(0, 1) < p
    x = edge_index.clone()
    x[:, drop_mask] = 0
    return x


def drop_edge_weighted(edge_index, edge_weights, p: float = 0.3, threshold: float = 0.7):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


class Contrast(nn.Module):
    def __init__(self, args, model, rec_data):
        super(Contrast, self).__init__()
        self.model : KLMCR = model
        self.device = args['device']
        self.tau = args['kgc_temperatue']
        self.kg_p_drop = args['kg_p_drop']
        self.ui_p_drop = args['ui_p_drop']
        self.mix_ratio = args['mix_ratio']
        self.num_users = self.model.num_users
        self.num_items = self.model.num_items
        self.item_np = rec_data.item_np
        self.user_np = rec_data.user_np
        self.ent2id = args['ent2id']
        self.confidence_drop = args['confidence_random_drop']
        
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

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
        view1, rel_v1 = self.drop_edge_random(
            h2t, h2r, self.kg_p_drop, self.model.num_entities)
        view2, rel_v2 = self.drop_edge_random(
            h2t, h2r, self.kg_p_drop, self.model.num_entities)
        #return view1, rel_v1, view2, rel_v2
        view_ii, rel_ii = self.drop_LLM_rectify(
            h2t, h2r, rectify_info, 'ii', self.model.num_entities, self.model.num_relations)
        view_ui, rel_ui = self.drop_LLM_rectify(
            h2t, h2r, rectify_info, 'ui', self.model.num_entities, self.model.num_relations)
        # view1, rel_v1 = self.drop_edge_random_rectify(
        #     h2t, h2r, rectify_info, 'ii', self.kg_p_drop, self.model.num_entities, self.model.num_relations)
        # view2, rel_v2 = self.drop_edge_random_rectify(
        #     h2t, h2r, rectify_info, 'ui', self.kg_p_drop, self.model.num_entities, self.model.num_relations)

        collect_info = {
            'view_ii': view_ii,
            'rel_ii': rel_ii,
            'view_ui': view_ui,
            'rel_ui': rel_ui,
            'view1' : view1,
            'relv1' : rel_v1,
            'view2' : view2,
            'relv2' : rel_v2
        }
        #return view_ii, rel_ii, view_ui, rel_ui
        return collect_info

    def get_ui_views_weighted(self, item_stabilities):
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
        g_weighted = self.ui_drop_weighted(item_mask)
        g_weighted.requires_grad = False
        return g_weighted

    def item_kg_stability(self, view1, relv1, view2, relv2):
        kgv1_ro = self.model.cal_item_embedding_from_kg(view1, relv1)
        kgv2_ro = self.model.cal_item_embedding_from_kg(view2, relv2)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return sim
    
    def ui_drop_weighted(self, item_mask):
        # item_mask: [item_num]
        item_mask = item_mask.tolist()
        n_nodes = self.num_users + self.num_items
        # [interaction_num]
        item_np = self.item_np
        keep_idx = list()
            # overall sample rate = 0.4*0.9 = 0.36
        for i, j in enumerate(item_np.tolist()):
            if item_mask[j - self.num_users] and random() > 0.6:
                keep_idx.append(i)
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

    def get_views(self, rectify_info, aug_side="both"):
        # drop (epoch based)
        # kg drop -> 2 views -> view similarity for item
        if aug_side == "ui":
            kgv1, kgv2 = None, None
        else:
            #kgv1, relv1, kgv2, relv2 = self.get_kg_views(rectify_info)
            collect_info = self.get_kg_views(rectify_info)
            v_ii, rel_ii, v_ui, rel_ui = collect_info['view_ii'], collect_info['rel_ii'], collect_info['view_ui'], collect_info['rel_ui']
            kgvii, relvii, kgvui, relvui = collect_info['view1'], collect_info['relv1'], collect_info['view2'], collect_info['relv2']

        if aug_side == "kg":
            uiv1, uiv2 = None, None
        else:
            stability = self.item_kg_stability(kgvii, relvii, kgvui, relvui).to(self.device)
            uiv1 = self.get_ui_views_weighted(stability)
            uiv2 = self.get_ui_views_weighted(stability)
            # stability_adjusted = self.item_kg_stability(v_ii, rel_ii, v_ui, rel_ui).to(self.device)
            # uiv3 = self.get_ui_views_weighted(stability_adjusted)
            # uiv4 = self.get_ui_views_weighted(stability_adjusted)
            #uiv1, uiv2 = self.transform_origin_graph(self.model.Graph), self.transform_origin_graph(self.model.Graph)

        contrast_views = {
            "kgvii": kgvii,
            "kgvui": kgvui,
            "uiv1": uiv1,
            "uiv2": uiv2,
            'relvii': relvui,
            'relvui': relvii,
            # 'uiv3': uiv3,
            # 'uiv4': uiv4,
            # 'v_ii': v_ii,
            # 'v_ui': v_ui,
            # 'rel_ii': rel_ii,
            # 'rel_ui': rel_ui
        }
        return contrast_views
    
    def drop_edge_random(self, head2tail, head2rel, p_drop, padding):
        res = dict()
        rel = head2rel.copy()
        for item, es in head2tail.items():
            new_es = list()
            for e in es:
                if (random() > p_drop):
                    new_es.append(e)
                else:
                    new_es.append(padding)
            res[item] = torch.IntTensor(new_es).to(self.device)
            rel[item] = torch.IntTensor(rel[item]).to(self.device)
        return res, rel
    
    def drop_LLM_rectify(self, head2tail, head2rel, rectify_info, subgraph:str, padding_e, padding_r):
        tri2add = rectify_info[f'{subgraph}_add']
        tri2del = rectify_info[f'{subgraph}_del']
        res_t, res_r = head2tail.copy(), head2rel.copy()
        ## delete triples based on LLM
        for triple in tri2del:
            try:
                head, _, tail = triple
                if isinstance(head, int) and isinstance(tail, int):
                    if head not in res_t or tail not in res_t[head]:
                        continue
                    if random() > self.confidence_drop:
                        idx = res_t[head].index(tail)
                        res_t[head][idx] = padding_e
            except:
                continue
        ## add triples based on LLM
        for head, rel, tail in tri2add:
            if isinstance(head, int) and isinstance(tail, int):
                if head not in res_t or tail <= self.num_users + self.num_items: ## In case it is triples of other kinds, not i-a
                    continue
                if random() > self.confidence_drop and padding_e in res_t[head]:
                    idx = res_t[head].index(padding_e)
                    res_t[head][idx] = tail
                    res_r[head][idx] = rel - 1
        for key in res_t:
            res_t[key] = torch.IntTensor(res_t[key]).to(self.device)
            res_r[key] = torch.IntTensor(res_r[key]).to(self.device)
        return res_t, 

    def drop_edge_random_rectify(self, head2tail, head2rel, rectify_info, subgraph:str, p_drop, padding_e, padding_r):
        res_e = dict()
        res_r = head2rel.copy()
        ## Dropout randomly first
        for item, es in head2tail.items():
            new_es = list()
            for e in es:
                if (random() > p_drop):
                    new_es.append(e)
                else:
                    new_es.append(padding_e)
            res_e[item] = new_es
        
        ## rectify based on LLM
        tri2add = rectify_info[f'{subgraph}_add']
        tri2del = rectify_info[f'{subgraph}_del']
        ## delete triples based on LLM
        for triple in tri2del:
            try:
                head, _, tail = triple
                if isinstance(head, int) and isinstance(tail, int):
                    if head not in res_e or tail not in res_e[head]:
                        continue
                    if random() > self.confidence_drop:
                        idx = res_e[head].index(tail)
                        res_e[head][idx] = padding_e
            except:
                continue

        ## add triples based on LLM
        for head, rel, tail in tri2add:
            if isinstance(head, int) and isinstance(tail, int):
                if head not in res_e or tail <= self.num_users + self.num_items:
                    continue
                if random() > self.confidence_drop and padding_e in res_e[head]:
                    idx = res_e[head].index(padding_e)
                    res_e[head][idx] = tail
                    res_r[head][idx] = rel - 1
        for key in res_e:
            res_e[key] = torch.IntTensor(res_e[key]).to(self.device)
            res_r[key] = torch.IntTensor(res_r[key]).to(self.device)
        return res_e, res_r
        