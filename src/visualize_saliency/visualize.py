import click
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils.utils import check_paths_exist
from .visualizer import SaliencyMapVisualizer



@click.command()
@click.option("--model_name", type=str, default="textattack/roberta-base-SST-2", help="Huggingface model id")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--output_folder", type=str, default="../../output/")
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

    tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2")
    model.to(device)
    model.eval()
    model.zero_grad()

    viz = SaliencyMapVisualizer(model, tokenizer, device)

    with open(input_file) as f:
        inputs = f.readlines()
        for text in inputs:
            viz.visualize_attribution(text)
