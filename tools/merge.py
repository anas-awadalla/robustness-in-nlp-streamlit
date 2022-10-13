# # for all model_name and dataset in to_merge.csv file overwrite that row in extractive_question_answering.csv

import pandas as pd

to_merge = pd.read_csv('results/to_merge.csv')
extractive_question_answering = pd.read_csv('results/extractive_question_answering_final.csv')

# for model_name and dataset are in to_merge.csv file overwrite that row in extractive_question_answering.csv or create new row
for index, row in to_merge.iterrows():
    model_name = row['model_name'].lower()
    dataset = row['dataset'].lower()
    # if model_name and dataset are in extractive_question_answering.csv file overwrite that row do uncased case comparison
    if len(extractive_question_answering.loc[(extractive_question_answering['model_name'].str.lower() == model_name) & (extractive_question_answering['dataset'].str.lower() == dataset.lower())]) > 0:
        extractive_question_answering.loc[(extractive_question_answering['model_name'].str.lower() == model_name) & (extractive_question_answering['dataset'].str.lower() == dataset.lower())] = row.values
    else:
        extractive_question_answering = extractive_question_answering.append(row, ignore_index=True)

# check for duplicate rows
for index, row in extractive_question_answering.iterrows():
    if len(extractive_question_answering.loc[(extractive_question_answering['model_name'] == row['model_name']) & (extractive_question_answering['dataset'].str.lower() == row['dataset'].lower())]) > 1:
        print('Duplicate row: ', row)

# sort df
extractive_question_answering = extractive_question_answering.sort_values(by=['model_name', 'dataset'])
extractive_question_answering.to_csv('results/extractive_question_answering_final.csv', index=False)
