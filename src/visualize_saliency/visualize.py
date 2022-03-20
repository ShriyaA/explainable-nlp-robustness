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



@click.command()
@click.option("--model_name", type=str, default="textattack/roberta-base-SST-2", help="Huggingface model id")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--output_folder", type=str, default="./output/")
@click.option("--seed", type=int)
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

    with open(input_file) as f:
        reader = csv.reader(f)
        for row in reader:
            text = row[0]
            true_label = row[1]
            html_obj = viz.visualize_attribution(text, true_label)
            html_text = html_obj.data.encode("UTF-8")
            with open(os.path.join(output_folder, 'output.html'), 'ab') as f:
                f.write(html_text)

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
