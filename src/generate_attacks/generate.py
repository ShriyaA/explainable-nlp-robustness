from email.policy import default
import click
import csv
import torch

from functools import partial
from utils.utils import check_paths_exist
from generate_attacks.word_deletion import word_deletion
from generate_attacks.misspelling import misspelling
from generate_attacks.synonym_substitution import synonym_substitution
from attribution.attribution import Attribution
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@click.command()
@click.option("--attack_type", type=click.Choice(['word_deletion', 'misspelling', 'synonym_substitution']), required=True)
@click.option("--seed", type=int)
@click.option("--output_file", type=str, default="./output/attacks.csv")
@click.option("--pct_words_to_swap", type=float, default="0.1", help="Percentage of words to swap. Used only if attack_type is synonym_substitution")
@click.option("--transformations_per_example", type=int, default=1, help="Number of transformations to create for each input sentence.")
@click.option("--substitution_method", type=click.Choice(['wordnet', 'masked_lm']), default='masked_lm', help="Method to use for finding synonyms. Used only if attack_type is synonym_substitution")
@click.option("--model_name", type=str, default="textattack/roberta-base-SST-2", help="Huggingface model id")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--target_selection", type=click.Choice(['k_most_attributed', 'k_least_attributed', 'random']), default='k_most_attributed', help="Method to select words to modify")
@click.option("--target_selection_k", type=int, default=1, help="Number of words to select")
@click.option("--seed", type=int)
@click.argument('data_file')
def generate(**config):

    attack_type, output_file, data_file = config['attack_type'], config['output_file'], config['data_file']
    target_selection = config['target_selection']
    check_paths_exist(data_file)

    if target_selection == 'k_most_attributed' or target_selection == 'k_least_attributed':
        model_name = config['model_name']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        model.zero_grad()

        attributor = Attribution(model, tokenizer, device, config['explainability_method'])

    if attack_type == 'synonym_substitution':
        transformation = partial(synonym_substitution, config["substitution_method"], config["pct_words_to_swap"], config["transformations_per_example"])
    elif attack_type == 'word_deletion':
        transformation = partial(word_deletion)
    elif attack_type == 'misspelling':
        transformation = partial(misspelling)
    
    output = []

    with open(data_file) as f:
        reader = csv.reader(f)
        for row in reader:
            text = row[0]
            label = row[1]
            if target_selection == 'k_most_attributed' or target_selection == 'k_least_attributed':
                scores = attributor.get_attribution(text, label)
                target_indices = torch.topk(scores, config['target_selection_k'], largest=target_selection=='k_most_attributed').indices.tolist()
            else:
                target_indices = torch.randint(0, scores.shape[0], (config['target_selection_k'],))
            print(target_indices)
            attacks = transformation(text, label, target_indices)
            output.extend([(text, x, label) for x in attacks])

        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(output)




    
