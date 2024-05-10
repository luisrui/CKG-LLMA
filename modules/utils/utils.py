import yaml
import torch
import random
import numpy as np
import json

from collections import defaultdict
from torch_scatter import scatter_sum, scatter_max

def read_yaml(path):
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def set_random_seed(seed=2020):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def triples_transfer_to_graph(subgraphs: list):
    edge_index_list = []
    edge_type_list = []

    for subgraph in subgraphs:
        heads, rels, tails = zip(*subgraph)

        heads = np.array(heads)
        tails = np.array(tails)
        rels = np.array(rels)

        edge_index_sub = np.stack((heads, tails), axis=0)
        edge_index_sub = torch.from_numpy(edge_index_sub).long()
        edge_type_sub = torch.from_numpy(rels).long()

        edge_index_list.append(edge_index_sub)
        edge_type_list.append(edge_type_sub)

    return edge_index_list, edge_type_list

def inner_product(a, b):
    return torch.sum(a * b, dim=-1)

def sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 32)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def edge_softmax(edge_index, edge_attr):
    _, dst = edge_index
    unique_dst, inv_idx = torch.unique(dst, return_inverse=True)
    
    edge_attr = torch.exp(edge_attr)
    
    sum_att_per_dst = scatter_sum(edge_attr, inv_idx, dim=0, dim_size=unique_dst.size(0))
    
    edge_attr = edge_attr / sum_att_per_dst[inv_idx]
    
    return edge_attr

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def Translate_triple2text(triples, id2ent, id2rel):
    """
    triples: list of triples
    id2ent: dict, id to entity
    id2rel: dict, id to relation
    """
    triples_text = str([(id2ent[head], id2rel[rel], id2ent[tail]) for head, rel, tail in triples])
    triples_text = triples_text.replace(', ', ',')
    return triples_text

def Translate_modify2id(modify_json, ent2id, rel2id):
    delete_list = modify_json['delete']
    try:
        delete_list_id = [(ent2id[head], rel2id[rel], ent2id[tail]) for head, rel, tail in delete_list]
    except:
        delete_list_id = []
        for triple in delete_list:
            try:
                head, rel, tail = triple
                delete_list_id.append((ent2id[head], rel2id[rel], ent2id[tail]))
            except:
                delete_list_id.append(triple)

    add_list = modify_json['add']
    try:
        add_list_id = [(ent2id[head], rel2id[rel], ent2id[tail]) for head, rel, tail in add_list]
    except:
        add_list_id = []
        for triple in add_list:
            try:
                head, rel, tail = triple
                add_list_id.append((ent2id[head], rel2id[rel], ent2id[tail]))
            except:
                add_list_id.append(triple)
    
    return {
        'delete': delete_list_id,
        'add': add_list_id
    }
    

def Read_prompt(initial_query : str, pos_items, neg_items, id2ent : dict, id2rel : dict, selected_triples : list):
    '''
    Generate graph prompts based on kg triples and logics.
    '''
    triples_text = Translate_triple2text(selected_triples, id2ent, id2rel)

    items = np.concatenate([pos_items, neg_items], axis=0)
    items = np.unique(items)
    id2name = json.load(open('./dataset/MovieLens1M/id2name.json', 'r'))
    items_title = {id2ent[item]:id2name[id2ent[item]] for item in items}
    items_title_text = str(items_title).replace(': ', ':').replace(', ',',')
    
    graph_prompt = initial_query.replace("<<Triples>>", triples_text).replace("<<TITLE_NAMES>>", items_title_text)
    return graph_prompt