import torch
from torch import nn
import torch.nn.functional as F

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
    
    # def regularization(self, h, t, r):
    #     regul = (torch.mean(h ** 2) + 
    #                 torch.mean(t ** 2) + 
    #                 torch.mean(r ** 2)) / 3
    #     return regul * regul
    
class TransR(nn.Module):
    '''
    TransR Model

    Compute the score of the triplets based on:
        score = ||P_r * (h + r) - t||_p
	'''
    def __init__(self, dim_e, dim_r, rel_num, p_norm = 1, norm_flag = True, rand_init = False):
        super(TransR, self).__init__()
        
        self.rel_num = rel_num
        
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.rand_init = rand_init

        self.transfer_matrix = nn.Embedding(self.rel_num, self.dim_e * self.dim_r)
        if not self.rand_init:
            identity = torch.zeros(self.dim_e, self.dim_r)
            for i in range(min(self.dim_e, self.dim_r)):
                identity[i][i] = 1
            identity = identity.view(self.dim_r * self.dim_e)
            for i in range(self.rel_num):
                self.transfer_matrix.weight.data[i] = identity
        else:
            nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def _transfer(self, e, r_transfer):
        r_transfer = r_transfer.view(-1, self.dim_e, self.dim_r)
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], self.dim_e).permute(1, 0, 2)
            e = torch.matmul(e, r_transfer).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.dim_e)
            e = torch.matmul(e, r_transfer)
        return e.view(-1, self.dim_r)

    def forward(self, head, tail, rel, rel_id, pos_num, mode='normal'):
        r_transfer = self.transfer_matrix(rel_id)
        head = self._transfer(head, r_transfer)
        tail = self._transfer(tail, r_transfer)
        score = self._calc(head ,tail, rel, mode)
        pos_score = self._get_positive_score(score, pos_num)
        neg_score = self._get_negative_score(score, pos_num)
        return pos_score, neg_score

    # def regularization(self, data):
    #     batch_h = data['batch_h']
    #     batch_t = data['batch_t']
    #     batch_r = data['batch_r']
    #     h = self.ent_embeddings(batch_h)
    #     t = self.ent_embeddings(batch_t)
    #     r = self.rel_embeddings(batch_r)
    #     r_transfer = self.transfer_matrix(batch_r)
    #     regul = (torch.mean(h ** 2) + 
    #                 torch.mean(t ** 2) + 
    #                 torch.mean(r ** 2) +
    #                 torch.mean(r_transfer ** 2)) / 4
    #     return regul * regul

    def predict(self, data):
        score = self.forward(data)
        return score
    
    def _get_positive_score(self, score, num_pos_samples):
        positive_score = score[:num_pos_samples]
        positive_score = positive_score.view(-1, num_pos_samples).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score, num_pos_samples):
        negative_score = score[num_pos_samples:]
        negative_score = negative_score.view(-1, num_pos_samples).permute(1, 0)
        return negative_score
