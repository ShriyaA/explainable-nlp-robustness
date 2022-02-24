import click
from .visualize_saliency.visualize import visualize

@click.group()
def main():
    pass

main.add_command()

if __name__ == '__main__':
    main(visualize, "visualize")
