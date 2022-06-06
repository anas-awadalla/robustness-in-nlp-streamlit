import pandas as pd

df = pd.read_csv('results/extractive_question_answering.csv')

# remove all models with model_name spanbert-base-cased-few-shot-k-{}-finetuned-squad-seed-{0,2,4} or 
# bert-base-uncased-few-shot-k-{}-finetuned-squad-seed-{0,2,4}
# roberta-base-few-shot-k-{}-finetuned-squad-seed-{0,2,4}
i = 0
while i < len(df):
    if ("spanbert-base-cased-few-shot-k-" in df.iloc[i]['model_name'] or "bert-base-uncased-few-shot-k-" in df.iloc[i]['model_name'] or "roberta-base-few-shot-k-" in df.iloc[i]['model_name']) and ("seed-0" in df.iloc[i]['model_name'] or "seed-2" in df.iloc[i]['model_name'] or "seed-4" in df.iloc[i]['model_name']):
        df = df.drop(df.index[i])
        print("removed")
    else:
        i += 1

df.to_csv('results/extractive_question_answering.csv', index=False)