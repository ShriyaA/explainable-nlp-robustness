import click
import csv

from tqdm import tqdm
from utils.utils import check_paths_exist
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import math

@click.command()
@click.option("--input_file", type=str, default="./output/search.csv")
@click.option("--output_file_1", type=str, default="./output/plot-cumulative.png")
@click.option("--output_file_2", type=str, default="./output/plot-buckets.png")
@click.option("--scoring_method", type=str, default="cosine")
def plotting(**config):
    
    input_file, output_file_1, output_file_2, scoring_method = config['input_file'], config['output_file_1'], config['output_file_2'], config['scoring_method']
    check_paths_exist(input_file)

    output = [('original_text', 'attacked_text', 'true_label', 'predicted_label', 'score')]
    output = [('original_text', 'best_attack', 'true_label', 'predicted_label', 'score')]

    with open(input_file) as f:
        reader = csv.reader(f)
        scores = [float(row[4]) for row in tqdm(reader) if row[4]!="score"]
    plot_scores_cumulative(scores, output_file_1, scoring_method)
    plot_scores_bucket(scores, output_file_2, scoring_method)


def plot_scores_cumulative(scores, file_name, scoring_method="cosine"):
    x_points = None
    successful_attacks = []

    if scoring_method == 'cosine':
        x_points = [x / 10.0 for x in range(-10, 11)]
    elif scoring_method == 'l_inf':
        scores = [1/score for score in scores if score != np.inf]
        right, left = max(scores), min(scores)
        right = math.ceil(right)
        left = math.floor(left)
        x_points = [x / 10.0 for x in range((left*10), (right*10)+1)]

    for threshold in x_points:
        mask = [i for i in scores if i<=threshold]
        successful_attacks.append(len(mask))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.bar([str(i) for i in x_points],successful_attacks,align='center', width=0.6)
    
    if scoring_method == "cosine":
        ax.set_xlabel("cosine threshold")
    elif scoring_method == "l_inf":
        ax.set_xlabel("l-inf inverse threshold")

    ax.set_ylabel("successful attacks")

    # Adding to increase spacing between x labels.
    # Taken from https://stackoverflow.com/questions/44863375/how-to-change-spacing-between-ticks-in-matplotlib
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.5 # inch margin
    s = maxsize/plt.gcf().dpi*len(x_points)+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])


    plt.savefig(file_name)


def plot_scores_bucket(scores, file_name, scoring_method = 'cosine'):
    '''Dinesh's plotting code'''
    keys = None
    if scoring_method == 'cosine':
        keys = [x / 10.0 for x in range(-10, 11)]
    elif scoring_method == 'l_inf':
        scores = [1/score for score in scores if score != np.inf]
        right, left = max(scores), min(scores)
        right = math.ceil(right)
        left = math.floor(left)
        keys = [x / 10.0 for x in range((left*10), (right*10)+1)]

    scores = [round(item, 1) for item in scores]
    counts = Counter(scores)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    frequencies = []
    for key in keys:
        if key in counts:
            frequencies.append(counts[key])
        else:
            frequencies.append(0)

    x_coordinates = list(range(len(keys)))
    ax.bar(x_coordinates, frequencies, align='center', width=0.6)
    
    if scoring_method == "cosine":
        ax.set_xlabel("cosine threshold")
    elif scoring_method == "l_inf":
        ax.set_xlabel("l-inf inverse threshold")

    ax.set_ylabel("successful attacks")

    ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(keys))

    # Adding to increase spacing between x labels.
    # Taken from https://stackoverflow.com/questions/44863375/how-to-change-spacing-between-ticks-in-matplotlib
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.5 # inch margin
    s = maxsize/plt.gcf().dpi*len(keys)+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.savefig(file_name)
