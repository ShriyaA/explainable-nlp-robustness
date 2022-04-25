import click
import csv

from tqdm import tqdm
from utils.utils import check_paths_exist
import matplotlib.pyplot as plt
from collections import Counter

@click.command()
@click.option("--input_file", type=str, default="./output/scores.csv")
@click.option("--output_file_1", type=str, default="./output/plot-cumulative.png")
@click.option("--output_file_2", type=str, default="./output/plot-buckets.png")
def plotting(**config):
    
    input_file, output_file_1, output_file_2 = config['input_file'], config['output_file_1'], config['output_file_2']
    check_paths_exist(input_file)

    output = [('original_text', 'attacked_text', 'true_label', 'predicted_label', 'score')]

    num_lines = 0
    with open(input_file) as f:
        num_lines = sum(1 for line in f)

    with tqdm(total=num_lines) as pbar:
        with open(input_file) as f:
            reader = csv.reader(f)
            scores = [float(row[4]) for row in tqdm(reader) if row[2]==row[3]]

    plot_scores_bucket(scores, output_file_1)
    plot_scores_cumulative(scores, output_file_2)


def plot_scores_cumulative(scores, file_name):
    cosine_threshold = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    successful_attacks = []
    for threshold in cosine_threshold:
        mask = [i for i in scores if i<=threshold]
        successful_attacks.append(len(mask))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.bar([str(i) for i in cosine_threshold],successful_attacks,align='center')
    ax.set_xlabel("cosine threshold")
    ax.set_ylabel("successful attacks")
    plt.savefig(file_name)

def plot_scores_bucket(scores, file_name):
    scores = [round(item, 1) for item in scores]
    counts = Counter(scores)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    keys = [x / 10.0 for x in range(0, 11)]
    frequencies = []
    for key in keys:
        if key in counts:
            frequencies.append(counts[key])
        else:
            frequencies.append(0)

    x_coordinates = list(range(len(keys)))
    ax.bar(x_coordinates, frequencies, align='center')

    ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(keys))
    ax.set_xlabel("cosine threshold")
    ax.set_ylabel("successful attacks")
    plt.savefig(file_name)