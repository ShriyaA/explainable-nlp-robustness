import imp
from textattack.transformations import WordSwapWordNet, WordSwapMaskedLM
from textattack.augmentation import Augmenter
from textattack.constraints.pre_transformation import StopwordModification
from textattack.constraints.grammaticality import PartOfSpeech

def synonym_substitution(substitution_method, pct_words_to_swap, transformations_per_example, sample, true_label):
    if substitution_method == 'wordnet':
        transformation = WordSwapWordNet()
    else:
        transformation = WordSwapMaskedLM()

    augmenter = Augmenter(transformation=transformation, constraints=[StopwordModification(), PartOfSpeech()], pct_words_to_swap=pct_words_to_swap, transformations_per_example=transformations_per_example)
    return augmenter.augment(sample)