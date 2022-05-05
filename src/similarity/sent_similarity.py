import click
import csv

from tqdm import tqdm
from utils.utils import check_paths_exist
from collections import Counter
from sentence_transformers import SentenceTransformer, util

@click.command()
@click.option("--input_file", type=str, default="./output/search.csv")
@click.option("--output_file", type=str, default="./output/scores-sim.csv")
def sent_similarity(**config):
	
	input_file, output_file = config['input_file'], config['output_file']
	check_paths_exist(input_file)

	with open(output_file, 'a') as f:
		writer = csv.writer(f)
		writer.writerow(['original_text','best_attack','true_label','predicted_label','score','attack_type','affected_indices','sent_score'])
	
	data=[]
	num_lines = 0
	with open(input_file) as f:
		num_lines = sum(1 for line in f)

	model = SentenceTransformer('all-mpnet-base-v2')

	with tqdm(total=num_lines) as pbar:
		with open(input_file) as f:
			reader = csv.reader(f)
			for row in tqdm(reader):
				if row[0]!="original_text":
					data.append(row+[str(get_BERT_similarity(model,row[0],row[1]))])
				pbar.update(1)

	with open(output_file, 'a') as f:
		writer = csv.writer(f)
		for row in data:
			writer.writerow(row)

def get_BERT_similarity(model, sent1, sent2):
	emb1 = model.encode(sent1)
	emb2 = model.encode(sent2)
	sim = float(util.cos_sim(emb1, emb2))
	return round(sim,2)