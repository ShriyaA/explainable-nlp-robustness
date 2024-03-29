from distutils.command.config import LANG_EXT
from email.policy import default
import imp
from multiprocessing.spawn import import_main_path
from xmlrpc.client import boolean
import click
import torch
import csv
import string

from functools import partial
from tqdm import tqdm
from utils.utils import check_paths_exist
from generate_attacks.word_deletion import word_deletion
from generate_attacks.misspelling import misspelling
from generate_attacks.synonym_substitution import synonym_substitution
from generate_attacks.word_inflection import word_inflection
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from attribution.attribution import Attribution
from evaluation import scoring

def clean_text(text):
    text = text.replace(" '",'')
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text = " ".join(text.split())
    return text

def extend_scores(scores, deleted_indexes):
    '''
    Adds 0s to the scores in place of deleted indexes.
    '''
    new_scores = []
    deleted_indexes = set(deleted_indexes)
    new_idx = 0
    score_idx = 0
    
    while score_idx < len(scores) or len(deleted_indexes) != 0:
        if new_idx in deleted_indexes:
            new_scores.append(0)
            deleted_indexes.remove(new_idx)
        else:
            new_scores.append(scores[score_idx])
            score_idx += 1
        
        new_idx += 1

    return torch.tensor(new_scores)

@click.command()
@click.option('--pct_candidates', type=float, default=0.3)
@click.option('--max_candidates', type=int, default=5)
@click.option('--substitution_method', type=click.Choice(['wordnet', 'embedding', 'masked-lm']))
@click.option('--target_selection', type=click.Choice(['most', 'least', 'random']), default='most')
@click.option("--evaluation_method", type=click.Choice(['cosine', 'l_inf']), default='cosine')
@click.option("--stopping_threshold", type=float, default=0.5)
@click.option("--attack_type", type=click.Choice(['word_deletion', 'misspelling', 'synonym_substitution', 'word_inflection', 'composite']), required=True)
@click.option("--model_name", type=str, default="textattack/roberta-base-SST-2", help="Huggingface model id")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--output_file", type=str, default="./output/attacks.csv")
@click.option('--clean_text', type=bool, default=False)
@click.option("--combination_method", type=click.Choice(['sum', 'max', 'avg']), default='avg')
@click.argument("data_file")
def greedy_search(**config):

    attack_type, output_file, data_file = config['attack_type'], config['output_file'], config['data_file']
    check_paths_exist(data_file)

    model_name = config['model_name']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.zero_grad()

    if attack_type == 'synonym_substitution':
        transformation = partial(synonym_substitution, config['substitution_method'], config['max_candidates'])
    elif attack_type == 'word_deletion':
        transformation = partial(word_deletion)
    elif attack_type == 'misspelling':
        transformation = partial(misspelling)
    elif attack_type == 'word_inflection':
        transformation = partial(word_inflection)

    attributor = Attribution(model, tokenizer, device, config['explainability_method'])
    output = ['original_text', 'best_attack', 'true_label', 'predicted_label', 'score', 'attack_type', 'affected_indices']
    scoring_func = getattr(scoring, config['evaluation_method'])

    num_lines = 0
    with open(data_file) as f:
        num_lines = sum(1 for line in f)

    with open(output_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(output)

    with tqdm(total=num_lines) as pbar:
        with open(data_file) as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                if config['clean_text']:
                    text = clean_text(text)
                
                label = int(row[1])
                attr_scores, tokenized_text, _ = attributor.get_attribution(text, label, word_level=True)
                
                num_candidates = int(config['pct_candidates'] * attr_scores.shape[0])
                target_indices = torch.topk(attr_scores, num_candidates, largest=config['target_selection']=='most').indices.tolist()
                target_indices = set(target_indices)
                similarity_score = float('inf')
                curr_deleted_idx = []
                curr_text = text
                curr_pred =  torch.tensor(label)

                min_tokenized_text = tokenized_text
                while similarity_score > config["stopping_threshold"] and (len(target_indices) != 0):
                    min_idx = -1
                    min_score = float('inf')
                    min_text = None
                    min_pred = None
                    no_viable_attack_indices = []
                    if attack_type != 'word_deletion':
                        tokenized_text = min_tokenized_text
                    
                    for idx in target_indices:
                    
                        if attack_type == 'word_deletion':
                            new_attacks = transformation(tokenized_text, curr_deleted_idx + [idx])
                        else:
                            new_attacks = transformation(tokenized_text, [idx])

                        if len(new_attacks) == 0:
                            no_viable_attack_indices.append(idx)
                            continue

                        new_attack = new_attacks[0]

                        if attack_type == 'synonym_substitution' or attack_type == 'word_inflection':
                            current_word_min_score = float('inf')
                            best_attack = new_attacks[0]
                            for i,attack in enumerate(new_attacks):
                                current_word_new_attr, _, current_word_pred = attributor.get_attribution(attack, label, word_level=True, combination_method=config['combination_method'])
                                current_word_score = scoring_func(attr_scores, current_word_new_attr)
                                if current_word_score < current_word_min_score:
                                    current_word_min_score = current_word_score
                                    best_attack = new_attacks[i]
                            new_attack = best_attack
                        
                        new_attr, curr_tokenized_text, pred = attributor.get_attribution(new_attack, label, word_level=True, combination_method=config['combination_method'])
                        if pred != label:
                            no_viable_attack_indices.append(idx)
                            continue
                        
                        if attack_type == 'word_deletion':
                            new_attr = extend_scores(new_attr, curr_deleted_idx + [idx])
                        
                        curr_score = scoring_func(attr_scores, new_attr)
                        if curr_score < min_score:
                            min_idx = idx
                            min_text = new_attack
                            min_score = curr_score
                            min_pred = pred
                            min_tokenized_text = curr_tokenized_text
                    
                    if len(no_viable_attack_indices) == len(target_indices):
                        break

                    curr_deleted_idx.append(min_idx)
                    target_indices.remove(min_idx)
                    similarity_score = min_score
                    curr_text = min_text
                    curr_pred = min_pred

                with open(output_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([text, curr_text, label, curr_pred.item(), round(similarity_score, 4), config['attack_type'], curr_deleted_idx])
                
                pbar.update(1)
