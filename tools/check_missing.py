# load extractive_question_answering.csv and check if there are some model_name entries missing datasets
import pandas as pd

df = pd.read_csv('results/extractive_question_answering_new.csv')

# Create a missing.csv file that contains the missing model_name entries and the corresponding dataset

with open('results/missing.csv', 'w') as f:
    # Write the header
    f.write('model_name,dataset\n')
    for model_name in df['model_name'].unique():
        df_model = df[df['model_name'] == model_name]
        for dataset in df['dataset'].str.lower().unique():
            # print('\t', dataset)
            df_model_dataset = df_model[df_model['dataset'].str.lower() == dataset]
            if len(df_model_dataset) == 0:
                f.write(model_name + ',' + dataset + '\n')