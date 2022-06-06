# load pandas df and remove everything before / in model_name column

import pandas as pd

df = pd.read_csv('results/to_merge.csv')

for i in range(len(df)):
    df.loc[i, 'model_name'] = df.loc[i, 'model_name'].split('/')[-1]
    
df.to_csv('results/to_merge.csv', index=False)