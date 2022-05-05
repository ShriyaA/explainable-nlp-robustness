from textattack.transformations import WordSwapWordNet, WordSwapMaskedLM, WordSwapEmbedding
from textattack.shared.attacked_text import AttackedText
from lemminflect import getLemma
from flair.data import Sentence
from textattack.shared.utils import flair_tag
from generate_attacks.word_inflection import TokenizerForFlair

def synonym_substitution(substitution_method, max_candidates, sample, indices_to_swap):
    if substitution_method == 'wordnet':
        transformation = WordSwapWordNet()
    elif substitution_method == 'embedding':
        transformation = WordSwapEmbedding(max_candidates=max_candidates*2) # Get double as some will be filtered out if they are inflections
    elif substitution_method == 'masked-lm':
        transformation = WordSwapMaskedLM(max_candidates=max_candidates)
    else:
        raise NotImplementedError()
    text = AttackedText(' '.join(sample))
    try:
        result = transformation._get_transformations(text, indices_to_swap[0])
    except:
        result = []
    result = filter_inflections(sample, result, indices_to_swap[0])
    result = [x.text for x in result]
    return result[:min(len(result), max_candidates)]

def filter_inflections(sample, result, index_swapped):
    orig_word = sample[index_swapped]
    sent = Sentence(" ".join(sample), use_tokenizer=TokenizerForFlair())
    flair_tag(sent)
    pos_tags = [token.annotation_layers["pos"][0]._value for token in sent]
    orig_lemma = getLemma(orig_word, pos_tags[index_swapped])
    if len(orig_lemma)==0:
        return result
    orig_lemma = orig_lemma[0]
    
    result_filtered = []

    for x in result:
        new_word = x.text.split(' ')[index_swapped]
        sent = Sentence(x.text, use_tokenizer=TokenizerForFlair())
        flair_tag(sent)
        pos_tags = [token.annotation_layers["pos"][0]._value for token in sent]
        if pos_tags[index_swapped] not in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'AUX']:
            result_filtered.append(x)
            continue
        new_lemma = getLemma(new_word, pos_tags[index_swapped])
        if len(new_lemma)==0:
            result_filtered.append(x)
            continue
        new_lemma = new_lemma[0]
        if new_lemma != orig_lemma:
            result_filtered.append(x)
    return result_filtered



if __name__=='__main__':
    print(synonym_substitution('embedding', 5,["The", "movie", "is", "great"], [1]))
    #print(synonym_substitution('masked-lm', 5,"The movie is great", [1]))