from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def plot_scores(scores):
    '''Dinesh's plotting code'''
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
    plt.savefig('graph.png')

if __name__=='__main__':
    data = pd.read_csv('./output/scores.csv', header=0)
    data = data[data['true_label'] == data['predicted_label']]
    scores = data['score'].to_list()
    plot_scores(scores)
