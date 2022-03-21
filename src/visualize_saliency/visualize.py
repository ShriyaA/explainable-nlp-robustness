import random
import csv
import os
import click
import torch
import torch
import string
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.utils import check_paths_exist
from .visualizer import SaliencyMapVisualizer
from visualize_saliency.word_attribution import word_attribution
from .calc_similarity_score import calc_similarity_score

def calc_similarity(orig_scores, orig_tokens, modified_scores, modified_tokens):
    orig = word_attribution(orig_tokens, orig_scores)
    modfied = word_attribution(modified_tokens, modified_scores)
    orig_scores = [item[0] for item in orig]
    orig_words = [item[2] for item in orig]
    modified_scores = [item[0] for item in modfied]
    modified_words = [item[2] for item in modfied]
    
    return calc_similarity_score(orig_scores, orig_words, modified_scores, modified_words)

@click.command()
@click.option("--model_name", type=str, default="textattack/roberta-base-SST-2", help="Huggingface model id")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--output_folder", type=str, default="./output/")
@click.option("--seed", type=int)
@click.option("--threshold", type=int, default=0.5)
@click.argument("input_file")
def visualize(**config):

    # Set a seed if it's not passed as an option
    if config["seed"] is None:
        config["seed"] = random.randint(0, 2 ** 32)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    model_name, expl_method, output_folder, input_file = config['model_name'], config['explainability_method'], config['output_folder'], config['input_file']

    check_paths_exist(output_folder, input_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.zero_grad()

    viz = SaliencyMapVisualizer(model, tokenizer, device, expl_method)

    num_lines = 0
    with open(input_file) as f:
        num_lines = sum(1 for line in f)
    
    with tqdm.tqdm(total=num_lines) as pbar:
        with open(input_file) as f:
            reader = csv.reader(f)
            with open(os.path.join(output_folder, 'output.html'), 'ab') as out:
                idx = 0
                for row in reader:
                    pbar.update(1)
                    text = row[0]
                    modified_text = row[1]
                    true_label = row[2]
                    html_obj, orig_scores, orig_tokens = viz.visualize_attribution(text, true_label)
                    orig_tokens = orig_tokens[1:][:-1] # Ignore start and end tokens
                    orig_scores = orig_scores[1:][:-1]
                    modified_html_obj, modified_scores, modified_tokens = viz.visualize_attribution(modified_text, true_label)
                    modified_tokens = modified_tokens[1:][:-1]
                    modified_scores = modified_scores[1:][:-1]
                    similarity = calc_similarity(orig_scores, orig_tokens, modified_scores, modified_tokens)
                    if similarity > config['threshold']:
                        continue

                    html_text = html_obj.data.encode("UTF-8")
                    modified_html_text = modified_html_obj.data.encode("UTF-8")
                    out.write(html_text)
                    out.write(modified_html_text)

def clean_text(text):
    text = text.replace(" '",'')
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text = " ".join(text.split())
    return text


def get_attribution_scores(model_name, expl_method, input_file, seed = None, remove_punctuation=True):
    '''
    returns:
    tokenizer
    scores    :  (token_ids, attribution_scores, true_label) for each line in the input_file.
    '''

    # Set a seed if it's not passed as an option
    if seed is None:
        seed = random.randint(0, 2 ** 32)
    torch.manual_seed(seed)
    random.seed(seed)

    check_paths_exist(input_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.zero_grad()

    viz = SaliencyMapVisualizer(model, tokenizer, device, expl_method)

    scores = []
    idx = 0
    num_lines = 0
    with open(input_file) as f:
        num_lines = sum(1 for line in f)
    
    with tqdm.tqdm(total=num_lines) as pbar:
        with open(input_file) as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                true_label = row[1]
                if remove_punctuation:
                    text = clean_text(text)
                scores.append(viz.get_attribution(text, true_label))
                idx += 1
                pbar.update(1)

    return tokenizer, scores
