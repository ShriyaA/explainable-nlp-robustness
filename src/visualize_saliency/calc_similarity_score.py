import math

def calc_similarity_score(orig_scores, orig_words, modified_scores, modified_words):
    orig_norm = math.sqrt(sum(i*i for i in orig_scores))
    modified_norm = math.sqrt(sum(i*i for i in modified_scores))
    norm_orig_scores = [score/orig_norm for score in orig_scores]
    norm_modified_scores = [score/modified_norm for score in modified_scores]

    orig_idx = 0
    mod_idx = 0
    ret = 0
    while orig_idx < len(orig_words) and mod_idx < len(modified_words):
        if orig_words[orig_idx] != modified_words[mod_idx]:
            orig_idx += 1
            continue

        ret += (norm_orig_scores[orig_idx]*norm_modified_scores[mod_idx])
        mod_idx += 1
        orig_idx += 1

    return ret
