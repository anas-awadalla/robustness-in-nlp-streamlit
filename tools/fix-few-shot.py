# read in pandas df and change type to fewshot if model_name had -k-16 or -k-32 or -k-64 or -k-128 or -k-256 or -k-512 or -k-1024 in it
import pandas as pd

df = pd.read_csv('results/extractive_question_answering_new.csv')

keywords = ['-k-16', '-k-32', '-k-64', '-k-128', '-k-256', '-k-512', '-k-1024']

for keyword in keywords:
    df.loc[df['model_name'].str.contains(keyword), 'type'] = 'fewshot'

df.to_csv('results/extractive_question_answering_new.csv', index=False)