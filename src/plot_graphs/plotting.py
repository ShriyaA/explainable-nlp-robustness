import click
import csv

from tqdm import tqdm
from utils.utils import check_paths_exist
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
import numpy as np
import pandas as pd
import math

@click.command()
@click.option("--input_file", type=str, default="./output/search.csv")
@click.option("--output_file_1", type=str, default="./output/plot-cumulative.png")
@click.option("--output_file_2", type=str, default="./output/plot-buckets.png")
@click.option("--additional_graphs/--no_additional_graphs", default=False)
@click.option("--scoring_method", type=str, default="cosine")
@click.option("--output_file_3", type=str, default="./output/plot-sent-sim.png")
@click.option("--color", type=str, default='C1')
def plotting(**config):
    
    input_file, output_file_1, output_file_2, scoring_method = config['input_file'], config['output_file_1'], config['output_file_2'], config['scoring_method']
    check_paths_exist(input_file)

    output = [('original_text', 'attacked_text', 'true_label', 'predicted_label', 'score')]
    output = [('original_text', 'best_attack', 'true_label', 'predicted_label', 'score')]

    mpl.style.use('seaborn-muted')

    with open(input_file) as f:
        reader = csv.reader(f)
        scores = [float(row[4]) for row in tqdm(reader) if row[4]!="score" and row[2]==row[3]]

    if config['additional_graphs']:
        df = pd.read_csv(input_file, converters={'affected_indices':lambda x: x.strip("[]").replace("'","").split(", ")})
        df = df[df['true_label']==df['predicted_label']]
        #df['len_affected_indices'] = df.affected_indices.apply(lambda x: len(x))
        df.plot.scatter(x='score', y='sent_score', alpha=0.5, color=config['color'])
        plt.axvline(x=0.5, color='C5')
        plt.axhline(y=0.7, color='C7')
        plt.xlabel("Cosine Similarity Score")
        plt.ylabel("Semantic Similarity Score")
        plt.savefig(config['output_file_3'])
        
    plot_scores_cumulative(scores, output_file_1, scoring_method, config['color'])
    plot_scores_bucket(scores, output_file_2, scoring_method, config['color'])


def plot_scores_cumulative(scores, file_name, scoring_method="cosine", color='C1'):
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
    
    ax.bar([str(i) for i in x_points],successful_attacks,align='center', width=0.6, color=color)
    
    if scoring_method == "cosine":
        ax.set_xlabel("Cosine Similarity Score")
    elif scoring_method == "l_inf":
        ax.set_xlabel("l-inf inverse threshold")

    ax.set_ylabel("Number of Samples")

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


def plot_scores_bucket(scores, file_name, scoring_method = 'cosine', color='tab:blue'):
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
    ax.bar(x_coordinates, frequencies, align='center', width=0.6, color=color)
    
    if scoring_method == "cosine":
        ax.set_xlabel("Cosine Similarity Score")
    elif scoring_method == "l_inf":
        ax.set_xlabel("l-inf inverse threshold")

    ax.set_ylabel("Number of Samples")

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

    plt.axvline(x=15.5, c='r', lw=1.5)
    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.savefig(file_name)
