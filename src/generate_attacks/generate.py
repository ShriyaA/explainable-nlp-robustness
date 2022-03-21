from email.policy import default
import click
import csv

from functools import partial
from utils.utils import check_paths_exist
# from textattack.transformations import WordSwapWordNet, WordSwapMaskedLM
# from textattack.augmentation import Augmenter
# from textattack.constraints.pre_transformation import StopwordModification
from generate_attacks.word_deletion import word_deletion
from generate_attacks.misspelling import misspelling

@click.command()
@click.option("--attack_type", type=click.Choice(['word_deletion', 'misspelling', 'synonym_substitution']), required=True)
@click.option("--seed", type=int)
@click.option("--output_file", type=str, default="./output/attacks.csv")
@click.option("--pct_words_to_swap", type=float, default="0.1", help="Percentage of words to swap. Used only if attack_type is synonym_substitution")
@click.option("--transformations_per_example", type=int, default=1, help="Number of transformations to create for each input sentence.")
@click.option("--substitution_method", type=click.Choice(['wordnet', 'masked_lm']), default='masked_lm', help="Method to use for finding synonyms. Used only if attack_type is synonym_substitution")
@click.option("--model_name", type=str, default="textattack/roberta-base-SST-2", help="Huggingface model id")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--seed", type=int)
@click.argument('data_file')
def generate(**config):

    attack_type, output_file, data_file = config['attack_type'], config['output_file'], config['data_file']

    check_paths_exist(data_file)

    if attack_type == 'synonym_substitution':
        transformation = partial(synonym_substitution, config["substitution_method"], config["pct_words_to_swap"], config["transformations_per_example"])
        output = []
        with open(data_file) as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                label = row[1]
                index = row[2]
                attacks = transformation(text)
                output.extend([(x, label, index) for x in attacks])

        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(output)
    elif attack_type == 'word_deletion':
        transformation = partial(word_deletion, config['data_file'], config['model_name'], config['explainability_method'], config['output_file'])
        transformation()
    elif attack_type == 'misspelling':
        transformation = partial(misspelling, config['data_file'], config['model_name'], config['explainability_method'], config['output_file'])
        transformation()
    else:
        raise NotImplementedError("Attack type not implemented")


def synonym_substitution(substitution_method, pct_words_to_swap, transformations_per_example, sample):
    if substitution_method == 'wordnet':
        transformation = WordSwapWordNet()
    else:
        transformation = WordSwapMaskedLM()

    augmenter = Augmenter(transformation=transformation, constraints=[StopwordModification()], pct_words_to_swap=pct_words_to_swap, transformations_per_example=transformations_per_example)
    return augmenter.augment(sample)
    
