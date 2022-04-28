import click
import csv

from tqdm import tqdm
from utils.utils import check_paths_exist
import matplotlib.pyplot as plt
from collections import Counter

@click.command()
@click.option("--input_file", type=str, default="./output/search.csv")
@click.option("--output_file", type=str, default="./output/scores-sim.csv")
def sent_similarity(**config):
	
	input_file, output_file = config['input_file'], config['output_file']
	check_paths_exist(input_file)

	with open(output_file, 'a') as f:
		writer = csv.writer(f)
		writer.writerow(['original_text', 'adversarial_text', 'label', 'expl_score', 'sent_score'])

	data=[]
	num_lines = 0
	with open(input_file) as f:
		num_lines = sum(1 for line in f)

	with tqdm(total=num_lines) as pbar:
		with open(input_file) as f:
			reader = csv.reader(f)
			for row in tqdm(reader):
				if row[0]!="original_text":
					data.append(row+[str(get_BERT_similarity(row[0],row[1]))])
				pbar.update(1)

	with open(output_file, 'a') as f:
		writer = csv.writer(f)
		for row in data:
			writer.writerow(row)

def get_BERT_similarity(sent1, sent2):
	return 0.5