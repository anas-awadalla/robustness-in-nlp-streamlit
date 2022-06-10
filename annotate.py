# A script that takes in a csv and prompts user for each unique model name to fill in the following properties
# 1. is_adapter
# 2. model_size in millions
# 3. is_robust_model

import pandas as pd
from tqdm import tqdm
import os

data = pd.read_csv('results/extractive_question_answering_new.csv')
annotated_data = pd.read_csv('results/annotated_data.csv') if os.path.exists('results/annotated_data.csv') else pd.DataFrame(columns=['model_name', 'is_adapter', 'model_size', 'is_robust'])

# Get unique model names minus the ones that are already annotated
models = set(data['model_name'].unique().tolist()).difference(set(annotated_data['model_name'].unique().tolist()))

# order the models
models = sorted(models)

# go through each unique model name
for model_name in tqdm(models):

    # prompt user for each of the following properties
    is_adapter = input('Is ' + model_name + ' an adapter? (y/n): ')
    if is_adapter == "exit":
        break
    is_adapter = is_adapter == "y"
    is_robust = input('Is ' + model_name + ' robust? (y/n): ') == 'y'

    model_size = None
    while not model_size:
        model_size = -1 # int(input('What is the model size of ' + model_name + ' in millions? '))

    # append the new row to the annotated data
    annotated_data = annotated_data.append({'model_name': model_name, 'is_adapter': is_adapter, 'model_size': model_size, 'is_robust': is_robust}, ignore_index=True)

# save the annotated data
annotated_data.to_csv('results/annotated_data.csv', index=False)
