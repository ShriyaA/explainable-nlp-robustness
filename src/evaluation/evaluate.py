import click
import csv
import torch

from tqdm import tqdm
from utils.utils import check_paths_exist
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from attribution.attribution import Attribution
from evaluation import scoring

@click.command()
@click.option("--output_file", type=str, default="./output/scores.csv")
@click.option("--evaluation_method", type=click.Choice(['cosine', 'l_inf']), default='cosine')
@click.option("--model_name", type=str, default="textattack/roberta-base-SST-2", help="Huggingface model id")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.argument('attack_file')
def evaluate(**config):
    
    output_file, attack_file = config['output_file'], config['attack_file']
    check_paths_exist(attack_file)

    model_name = config['model_name']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.zero_grad()

    attributor = Attribution(model, tokenizer, device, config['explainability_method'])

    scoring_func = getattr(scoring, config['evaluation_method'])

    output = [('original_text', 'attacked_text', 'true_label', 'predicted_label', 'score')]

    num_lines = 0
    with open(attack_file) as f:
        num_lines = sum(1 for line in f)

    with tqdm(total=num_lines) as pbar:
        with open(attack_file) as f:
            reader = csv.reader(f)
            for row in tqdm(reader):
                orig_text = row[0]
                attacked_text = row[1]
                true_label = row[2]

                orig_scores, _ = attributor.get_attribution(orig_text, true_label, word_level=True)
                attacked_scores, predicted_label = attributor.get_attribution(attacked_text, true_label, word_level=True)

                score = scoring_func(orig_scores, attacked_scores)

                output.append((orig_text, attacked_text, true_label, predicted_label.item(), score.item()))

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)