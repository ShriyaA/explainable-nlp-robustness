import click
from visualize_saliency.visualize import visualize
from generate_attacks.generate import generate

@click.group()
def main():
    pass

main.add_command(visualize)
main.add_command(generate)

if __name__ == '__main__':
    main()
