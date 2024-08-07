import yaml
import torch
import random
import os
import wandb
import uuid
import numpy as np
import json
from torch import nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_

from collections import defaultdict
from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from torch_scatter import scatter_sum
from copy import copy

class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = True
        config.prefix = "KGExplainer"
        config.project = "LLM_KG_Rec"
        config.output_dir = "./log"### load your own checkpoints file path here
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        config.entity = config_dict.placeholder(str)
        config.prefix_to_id = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, enable=True):
        self.enable = enable
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            if self.config.prefix_to_id:
                self.config.experiment_id = "{}--{}".format(
                    self.config.prefix, self.config.experiment_id
                )
            else:
                self.config.project = "{}--{}".format(self.config.prefix, self.config.project)

        if self.enable:
            if self.config.output_dir == "":
                raise 'no ourpur dir error!'
            else:
                self.config.output_dir = os.path.join(
                    self.config.output_dir, self.config.experiment_id
                )
                os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self._variant,
                project=self.config.project,
                dir=self.config.output_dir,
                id=self.config.experiment_id,
                anonymous=self.config.anonymous,
                notes=self.config.notes,
                entity=self.config.entity,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
                resume=True,
            )
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir
    
def read_yaml(path):
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def print_yaml(args : dict):
    for key, value in args.items():
        print(f"{key}: {value}")

def xavier_uniform_initialization(module):
    r"""using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

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
        if len(subgraph) != 0:
            heads, rels, tails = zip(*subgraph)

            heads = np.array(heads)
            tails = np.array(tails)
            rels = np.array(rels)

            edge_index_sub = np.stack((heads, tails), axis=0)
            edge_index_sub = torch.from_numpy(edge_index_sub).long()
            edge_type_sub = torch.from_numpy(rels).long()

            edge_index_list.append(edge_index_sub)
            edge_type_list.append(edge_type_sub)
        else:
            raise 'Problem in subgraph!'

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

def edge_softmax(heads, tails, edge_attr): # Ensure that the output edge is normalized(multi-attrs to one item)
    unique_dst, inv_idx = torch.unique(heads, return_inverse=True)
    
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
    
def Read_prompt(dataset:str, initial_query : str, pos_items, neg_items, id2ent : dict, id2rel : dict, selected_triples : list):
    '''
    Generate graph prompts based on kg triples and logics.
    '''
    triples_text = Translate_triple2text(selected_triples, id2ent, id2rel)

    items = np.concatenate([pos_items, neg_items], axis=0)
    items = np.unique(items)
    id2name = json.load(open(f'./dataset/{dataset}/id2name.json', 'r'))
    items_title = {id2ent[item]:id2name[id2ent[item]] for item in items}
    items_title_text = str(items_title).replace(': ', ':').replace(', ',',')
    
    graph_prompt = initial_query.replace("<<Triples>>", triples_text).replace("<<TITLE_NAMES>>", items_title_text)
    return graph_prompt

def Aug_graph(triples, num_users, num_items, ent2id, rel2id, aug_ratio, aug_type):
    """
    Augment the graph by adding negative samples.
    """
    triples = np.array(triples, dtype=int)
    if aug_type == 'node':
        item_attrs_startid = num_users + num_items
        item_attrs_ids = [i for i in range(item_attrs_startid, len(ent2id))]
        drop_item_attrs = np.random.choice(item_attrs_ids, size=int(len(item_attrs_ids) * aug_ratio), replace=False)

        augmented_triples = []
        for triple in triples:
            if not any(node_id in drop_item_attrs for node_id in triple):
                augmented_triples.append(triple)
        
        return augmented_triples
    elif aug_type == 'edge':
        augmented_triples = []   
        num_triples = len(triples)
        
        drop_mask = np.random.rand(num_triples) < aug_ratio
        liked_mask = triples[:, 1] == rel2id['liked']
        
        final_drop_mask = np.logical_and(drop_mask, liked_mask)
        augmented_triples = triples[~final_drop_mask]
        
        return augmented_triples

def Generate_rectify_info():
    return
    