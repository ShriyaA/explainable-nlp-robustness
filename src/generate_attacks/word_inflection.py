from lemminflect import getLemma, getAllInflections
from flair.data import Sentence, Tokenizer
from textattack.shared.utils import flair_tag

def word_inflection(sample, indices_to_swap):
    sent = Sentence(" ".join(sample), use_tokenizer=TokenizerForFlair())
    flair_tag(sent)
    pos_tags = [token.annotation_layers["pos"][0]._value for token in sent]
    attacks = []
    for idx in indices_to_swap:
        inflections = get_inflections(sample[idx], pos_tags[idx])
        for infl in inflections:
            attacks.append(' '.join(sample[:idx]+[infl]+sample[idx+1:]))
    #if len(attacks) == 0:
    #    attacks = [" ".join(sample)]
    return attacks
    
        
def get_inflections(word, pos_tag):
    if pos_tag not in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'AUX']:
        return []
    lemma = getLemma(word, pos_tag)
    if len(lemma) == 0:
        lemma = word
    inflections = getAllInflections(lemma[0])
    inflections = [inflections[k][0] for k in inflections if inflections[k][0] != word]
    return inflections

class TokenizerForFlair(Tokenizer):
    def tokenize(self, text: str):
        return text.split()

    @property
    def name(self):
        return self.__class__.__name__


if __name__=='__main__':
    print(word_inflection(["The", "movie", "is", "great"], [3]))