import os
import json
import torch
from torch import nn   
import math
import torch.nn.functional as F

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def load_checkpoint(self, path, device):
        self.load_state_dict(torch.load(os.path.join('./checkpoint', path), map_location=device))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), f=path)
    
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

class Sampler(object):
    """Base class for all sampler to sample negative items.
    """
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

class CrossAttentionLayer(nn.Module):
    def __init__(self, emb_size, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        assert self.head_dim * num_heads == emb_size, "emb_size must be divisible by num_heads"
        
        self.W_q = nn.Linear(emb_size, emb_size)
        self.W_k = nn.Linear(emb_size, emb_size)
        self.W_v = nn.Linear(emb_size, emb_size)

    def forward(self, query, key, value, mask=None):
        item_num, entity_num, _ = query.size()
        
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.emb_size)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -9e15)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        #attn_probs = attn_probs.permute(0, 2, 3, 1).contiguous().view(item_num, entity_num, self.num_heads)
        context = torch.matmul(attn_probs, v)
        
        context = context.transpose(1, 2).contiguous().view(item_num, entity_num, -1)
        return context
