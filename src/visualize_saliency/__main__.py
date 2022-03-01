import click

@click.command()
@click.option("--model-path", type=str, default="../../model/", help="Directory where model is available")
@click.option("--explainability-method", type=click.Choice(['IntegratedGradients'], case_sensitive=False), default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--output-folder", type=str, default="../../output/")
@click.argument("input_file")
def visualize():
    pass