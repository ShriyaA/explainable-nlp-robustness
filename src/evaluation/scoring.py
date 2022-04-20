from hashlib import new
from torch.nn import functional as F
import torch

def cosine(orig_scores, new_scores):
    return F.cosine_similarity(orig_scores, new_scores, dim=0).item()

def l_inf(orig_scores, new_scores):
    orig_max = torch.max(orig_scores)
    new_max = torch.max(new_scores)
    # returning inverse so that for both the cosine and l_inf scoring methods, the target is achieve low score.
    score = abs(orig_max.item() - new_max.item())
    return 1/(score + 1e-9)
