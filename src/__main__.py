import click
from visualize_saliency.visualize import visualize

@click.group()
def main():
    pass

main.add_command(visualize)

if __name__ == '__main__':
    main()
