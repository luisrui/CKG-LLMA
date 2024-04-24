import torch
from torch import nn
import torch.nn.functional as F

def l2_loss(*weights):
    """L2 loss

    Compute  the L2 norm of tensors without the `sqrt`:

        output = sum([sum(w ** 2) / 2 for w in weights])

    Args:
        *weights: Variable length weight list.

    """
    loss = 0.0

    for w in weights:
        loss += torch.sum(torch.pow(w, 2))

    average_loss = 0.5 * loss / len(weights) if len(weights) > 0 else 0
    return average_loss

class MarginLoss(nn.Module):
    '''Margin loss for ranking

    Computer the margin loss for a batch of data:

        output = max(0, margin - (postive_score - negative_score))
    
    Args:
        adv_temperature: The temperature of the adversarial loss. Default: None
    '''
    def __init__(self, adv_temperature = None, margin = 3.0):
        super(MarginLoss,self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
        else:
            return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
            
    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()