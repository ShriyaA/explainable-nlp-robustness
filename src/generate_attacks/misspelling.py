import click
import csv
import os
import pickle
from random import randrange

from utils.utils import check_paths_exist
from visualize_saliency.visualize import get_attribution_scores
from visualize_saliency.word_attribution import word_attribution

def get_misspelled_word(word, misspell_dict):
    
    if word in misspell_dict:
        return misspell_dict[word][0]
    
    index = randrange(len(word))
    
    return word[:index]+word[index+1:] 

def get_edited_string_mispell(tokenizer, score_word_list, tokens, misspell_dict, num_tokens_to_mispell = 1 ):

    remove_tokens_idx = []
    score_word_list = score_word_list[::-1]
    for i in range(num_tokens_to_mispell):
        remove_tokens_idx += score_word_list[i][1]

    remove_tokens_idx.sort()
    edited_tokens = []
    idx = 0
    for i in range(len(tokens)):
        if idx < len(remove_tokens_idx) and remove_tokens_idx[idx] == i:
            idx += 1
            replacement_word = get_misspelled_word(tokens[i][1:],misspell_dict)
            edited_tokens.append("Ġ"+replacement_word)
            continue
        edited_tokens.append(tokens[i])
    
    string = tokenizer.convert_tokens_to_string(edited_tokens)
    return string

def misspelling(input_file, model_name, expl_method, output_file, seed = None, remove_punctuation=True):
    out_sentences_1 = []
    
    misspellings_dict_path = "data/misspellings.pkl"
    check_paths_exist(misspellings_dict_path)
    with open(misspellings_dict_path, 'rb') as handle:
        misspellings_dict = pickle.load(handle)

    tokenizer, scores = get_attribution_scores(model_name, expl_method, input_file, seed, remove_punctuation)
    for input_ids, token_scores, true_label in scores:
        input_ids = input_ids[1:][:-1] # Ignore start and end tokens
        token_scores = token_scores[1:][:-1] # Ignore start and end token scores
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        score_word_list = word_attribution(tokens, token_scores)

        score_word_list.sort()
        
        orig = tokenizer.convert_tokens_to_string(tokens)
        edited1 = get_edited_string_mispell(tokenizer, score_word_list,  tokens, misspellings_dict, 1)
        out_sentences_1.append([orig, edited1, true_label])
        
    head, tail = os.path.splitext(output_file)
    # Write text with deleted words to output file
    with open(head+"_1"+tail, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(out_sentences_1)
