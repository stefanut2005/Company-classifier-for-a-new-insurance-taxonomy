# Company classifier for a new insurance taxonomy
## Overview
This Python script classifies companies based on their business description, niche, and business tags using zero-shot learning. It leverages the Hugging Face Transformers library with the RoBERTa-large-MNLI model to assign relevant labels from a predefined insurance taxonomy.

The script processes the dataset in batches of 8 for optimized performance and saves the classification results in a CSV file.

## Requirements
Make sure you have the following dependencies installed: `pip install pandas transformers torch`. Ensure that you have the following CSV files in your working directory: `ml_insurance_challenge.csv` → Contains company descriptions, business tags, and niche.
`insurance_taxonomy - insurance_taxonomy.csv` → Contains predefined labels for classification.

## Explanation
First of all, I thought of a brute-force solution, that was using the classifier and printing the best fitted labels based on the score, for each row in the table, but it was a very slow approach. Also, I tried a bunch of AI models (such as BERT, DistilBERT, DeBERTa, XLNet, T5, ALBERT) to find the most accurate and fastest one and I came to the conclusion that RoBERTa is the right one.

Knowing that the solution is not fast enough, I looked for an optimization and decided to use batch processing. Instead of processing one company at a time, the input dataset is divided into batches of 8 companies (or another configurable batch size). I measured the speeds of the two approaches to the challenge and concluded that batch processing is almost 2 times faster than the other approach, which is essential in processing a large amount of data.

So, at the beginning, the program loads the 2 files containing the data to be processed, initializes the classifier and the new column in the table. I divide the rows by 8 (batch_size) and for each of them I set the text to be analyzed: in case "business_tags" is empty, I use "description" and "niche", and if not, I use "business_tags" and "niche". Then, I apply the classifier and keep the first 10 best labels, which are later added to "insurance_table" and saved in our resulting file, "classified_companies.csv".
