import click
import csv
from visualize_saliency.visualize import get_attribution_scores
from visualize_saliency.word_attribution import word_attribution

def get_edited_string(tokenizer, score_word_list, tokens, num_words_to_remove = 1):
    '''
    returns text after deleting *num_words_to_remove* words
    '''
    remove_tokens_idx = []

    for i in range(num_words_to_remove):
        remove_tokens_idx += score_word_list[i][1]

    remove_tokens_idx.sort()
    edited_tokens = []
    idx = 0
    for i in range(len(tokens)):
        if idx < len(remove_tokens_idx) and remove_tokens_idx[idx] == i:
            idx += 1
            continue
        edited_tokens.append(tokens[i])

    string = tokenizer.convert_tokens_to_string(edited_tokens)
    return string

def word_deletion(input_file, model_name, expl_method, output_file, seed = None, remove_punctuation=True):
    out_sentences = []

    tokenizer, scores = get_attribution_scores(model_name, expl_method, input_file, seed, remove_punctuation)
    for input_ids, token_scores, true_label in scores:
        input_ids = input_ids[1:][:-1] # Ignore start and end tokens
        token_scores = token_scores[1:][:-1] # Ignore start and end token scores
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        score_word_list = word_attribution(tokens, token_scores)

        score_word_list.sort()
        if len(score_word_list) <= 2:
            continue

        orig = tokenizer.convert_tokens_to_string(tokens)
        edited1 = get_edited_string(tokenizer, score_word_list,  tokens, 1)
        edited2 = get_edited_string(tokenizer, score_word_list,  tokens, 2)
        out_sentences.append([orig, edited1, edited2, true_label])

    # Write text with deleted words to output file
    with open(output_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(out_sentences)
