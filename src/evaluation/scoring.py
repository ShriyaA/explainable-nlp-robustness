from hashlib import new
from torch.nn import functional as F
import torch

def cosine(orig_scores, new_scores):
    return F.cosine_similarity(orig_scores, new_scores, dim=0)

def l_inf(orig_scores, new_scores):
    orig_max = torch.max(orig_scores)
    new_max = torch.max(new_scores)
    return orig_max.item() - new_max.item()