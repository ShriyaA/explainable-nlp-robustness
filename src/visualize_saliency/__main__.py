import click

@click.command()
@click.option("--model_path", type=str, default="../../model/", help="Directory where model is available")
@click.option("--explainability_method", type=str, default="IntegratedGradients", help="Algorithm to use for generating saliency maps")
@click.option("--output_folder", type=str, default="../../output/")
@click.argument("input_file")
def visualize():
    pass