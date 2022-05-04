from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import math

def plot_scores(scores, scoring_method = 'cosine'):
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

    plt.savefig('graph.png')

if __name__=='__main__':
    data = pd.read_csv('./output/scores.csv', header=0)
    data = data[data['true_label'] == data['predicted_label']]
    scores = data['score'].to_list()
    plot_scores(scores, 'cosine')
