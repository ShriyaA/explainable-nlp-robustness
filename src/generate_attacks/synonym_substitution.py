from textattack.transformations import WordSwapWordNet, WordSwapMaskedLM
from textattack.augmentation import Augmenter

def synonym_substitution(sample, substitution_method='masked_lm', pct_words_to_swap=0.1, transformations_per_example=3):
    if substitution_method == 'wordnet':
        transformation = WordSwapWordNet()
    else:
        transformation = WordSwapMaskedLM()

    augmenter = Augmenter(transformation=transformation)
    return augmenter.augment(sample)