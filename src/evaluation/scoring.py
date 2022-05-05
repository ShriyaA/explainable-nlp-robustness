from hashlib import new
from torch.nn import functional as F
import torch

def cosine(orig_scores, new_scores):
    return F.cosine_similarity(orig_scores, new_scores, dim=0).item()

def l_inf(orig_scores, new_scores):
    diff = torch.sub(orig_scores, new_scores)
    absolute = torch.abs(diff)
    # returning inverse so that for both the cosine and l_inf scoring methods, the target is achieve low score.
    return 1/(torch.max(absolute).item() + 1e-6)