import pandas as pd
extractive_question_answering = pd.read_csv('results/extractive_question_answering_final.csv')
df_with_fewshot = extractive_question_answering[extractive_question_answering['type'] == 'fewshot']
df_with_fewshot["model_name"] = df_with_fewshot["model_name"].str.split("-seed-").str[0]
df_without_fewshot = extractive_question_answering[extractive_question_answering['type'] != 'fewshot']

# group by model_name and dataset and average f1 and em columns
print(df_with_fewshot)
df_with_fewshot = df_with_fewshot.groupby(['model_name', 'type', 'model_family', 'dataset']).mean().reset_index()
# add a _avg suffix to the model_name column entries
df_with_fewshot["model_name"] = df_with_fewshot["model_name"] + "-avg"

print(df_with_fewshot.head())
        
# merge the two dataframes
df = pd.concat([df_without_fewshot, df_with_fewshot])
# save as csv
df.to_csv('results/extractive_question_answering_final_avg.csv', index=False)