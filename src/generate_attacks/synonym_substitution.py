from textattack.transformations import WordSwapWordNet, WordSwapMaskedLM, WordSwapEmbedding
from textattack.augmentation import Augmenter
from textattack.constraints.pre_transformation import StopwordModification
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.shared.attacked_text import AttackedText

def synonym_substitution(substitution_method, max_candidates, sample, indices_to_swap):
    if substitution_method == 'wordnet':
        transformation = WordSwapWordNet()
    elif substitution_method == 'embedding':
        transformation = WordSwapEmbedding(max_candidates=max_candidates)
    elif substitution_method == 'masked-lm':
        transformation = WordSwapMaskedLM(max_candidates=max_candidates)
    else:
        raise NotImplementedError()
    sample = AttackedText(sample)
    result = transformation._get_transformations(sample, indices_to_swap)
    result = [x.text for x in result]
    return result


if __name__=='__main__':
    print(synonym_substitution('embedding', 5,"The movie is great", [1]))
    print(synonym_substitution('masked-lm', 5,"The movie is great", [1]))