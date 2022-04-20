import imp
from multiprocessing.spawn import import_main_path
import click
import torch
import csv

from functools import partial
from tqdm import tqdm
from utils.utils import check_paths_exist
from generate_attacks.word_deletion import word_deletion
from generate_attacks.misspelling import misspelling
from generate_attacks.synonym_substitution import synonym_substitution
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from attribution.attribution import Attribution
from evaluation import scoring

@click.command()
@click.option('--pct_candidates', type=float, default=0.3)
@click.option('--target_selection', type=click.Choice(['most', 'least', 'random']), default='most')
@click.option("--evaluation_method", type=click.Choice(['cosine', 'l_inf']), default='cosine')
@click.option("--stopping_threshold", type=float, default=0.5)
@click.option("--attack_type", type=click.Choice(['word_deletion', 'misspelling', 'synonym_substitution', 'composite']), required=True)
@click.option("--model_name", type=str, default="textattack/roberta-base-SST-2", help="Huggingface model id")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--output_file", type=str, default="./output/attacks.csv")
@click.argument("--data_file")
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
        transformation = partial(synonym_substitution)
    elif attack_type == 'word_deletion':
        transformation = partial(word_deletion)
    elif attack_type == 'misspelling':
        transformation = partial(misspelling)

    attributor = Attribution(model, tokenizer, device, config['explainability_method'])
    output = [('original_text', 'best_attack', 'true_label', 'score', 'attack_type', 'affected_indices')]
    scoring_func = getattr(scoring, config['evaluation_method'])

    num_lines = 0
    with open(data_file) as f:
        num_lines = sum(1 for line in f)

    with tqdm(total=num_lines) as pbar:
        with open(data_file) as f:
            reader = csv.reader(f)
            for row in tqdm(reader):
                text = row[0]
                label = row[1]
                attr_scores, _ = attributor.get_attribution(text, label, word_level=True)
                num_candidates = int(config['pct_candidates'] * attr_scores.shape[0])
                target_indices = torch.topk(attr_scores, num_candidates, largest=config['target_selection']=='most').indices.tolist()
                cosine_similarity = 1
                affected_indices = []
                current_attack = text

                while cosine_similarity > 0.5 and len(current_attack) > 0:
                    for idx in target_indices:
                        new_attack = transformation(current_attack, [idx])
                        new_attr, pred = attributor.get_attribution(new_attack, label, word_level=True)
                        if not pred == label:
                            continue
                        