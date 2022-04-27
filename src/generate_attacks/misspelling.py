import pickle
from random import randrange
from utils.utils import check_paths_exist

from textattack.transformations import WordSwapQWERTY
from textattack.augmentation import Augmenter

def get_misspelled_word(word, misspell_dict, augmenter):
    try:
        if word in misspell_dict:
            return misspell_dict[word][0]
        
        return augmenter.augment(word)[0]
    except:
        return word

def misspelling(text, target_indices, remove_punctuation=False):
    
    misspellings_dict_path = "data/misspellings.pkl"
    check_paths_exist(misspellings_dict_path)
    with open(misspellings_dict_path, 'rb') as handle:
        misspellings_dict = pickle.load(handle)

    tokens = text.split(' ')
    tokens = [x if i not in target_indices else get_misspelled_word(x, misspellings_dict) for i,x in enumerate(tokens)]
    edited = ' '.join(tokens)
    return [edited]
