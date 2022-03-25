# explainable-nlp-robustness

## Requirements

This project uses Python 3.8

## Setup Steps

1. Clone this repository  
		
		git clone https://github.com/ShriyaA/explainable-nlp-robustness.git
    
2. Create a virtual environment and activate it (Recommended)

		cd explainable-nlp-robustness
		python -m venv venv
		source venv/bin/activate
		
3. Install required packages

		pip install -r requirements.txt
		
4. Create output folder
		
		mkdir output
		
## Run with default options

	python src/__main__.py visualize input.txt
	
## Generate attacks

Word Deletion:

	python src/__main__.py generate data/glue-sst2-validation.csv --attack_type word_deletion

Misspelling:

	python src/__main__.py generate data/glue-sst2-validation.csv --attack_type misspelling
		
## Input format

The input file should be a two-column csv file with text sample in the first column and true label in the second column. Example:

	it 's a charming and often affecting journey .,1
	unflinchingly bleak and desperate,0

## Output format

The output is written to a file called 'output.html' in the output directory. Output is always written in append mode. The html file can be opened in the browser and shows the output in the same format created by captum's `visualize_text` function.
