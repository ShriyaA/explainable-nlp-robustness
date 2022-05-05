import click
from visualize_saliency.visualize import visualize
from generate_attacks.generate import generate
from evaluation.evaluate import evaluate
from search.greedy_search import greedy_search
from plot_graphs.plotting import plotting
from similarity.sent_similarity import sent_similarity

@click.group()
def main():
    pass

main.add_command(visualize)
main.add_command(generate)
main.add_command(evaluate)
main.add_command(greedy_search)
main.add_command(plotting)
main.add_command(sent_similarity)

if __name__ == '__main__':
    main()
