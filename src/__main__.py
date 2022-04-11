import click
from visualize_saliency.visualize import visualize
from generate_attacks.generate import generate
from evaluation.evaluate import evaluate

@click.group()
def main():
    pass

main.add_command(visualize)
main.add_command(generate)
main.add_command(evaluate)

if __name__ == '__main__':
    main()
