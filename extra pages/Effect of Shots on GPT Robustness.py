from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import streamlit as st
import plotly.express as px
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy
import random

st.set_page_config(
    page_title="Effect of Training Shots on Robustness",
    layout="wide",
)
    

dataset_to_size = {
    'squad': 10570,
    'squadshifts_nyt': 10065,
    'squadshifts_reddit': 9803,
    'squadshifts_new_wiki': 7938,
    'squadshifts_amazon': 9885,
    'RACE': 674, 
    'DROP': 1503, 
    'TextbookQA': 1503,
    'BioASQ': 1504,
    'RelationExtraction': 2948,
    'NewsQA': 4212,
    'DuoRC.ParaphraseRC': 1501,
    'TriviaQA-web': 7785,
    'SearchQA': 16980,
    'HotpotQA': 5904,
    'NaturalQuestionsShort': 12836
}

average_all_datasets = st.sidebar.checkbox("Average All Datasets", value=True, disabled=True)

dataset = st.sidebar.selectbox(
    'Dataset (Y-Axis)',
    list(dataset_to_size.keys()), disabled=average_all_datasets)

id_dataset = st.sidebar.selectbox(
    'Dataset (X-Axis)',
    list(dataset_to_size.keys()))


pandas_dataset = dataset
pandas_id_dataset = id_dataset

scaling = 'Linear'
color_by_dataset = False #st.sidebar.checkbox("Color by Pretraining Dataset", value=False)

results_path = Path(".") / "results"

df = pd.read_csv(os.path.join(results_path.absolute(),
                 'extractive_question_answering_new.csv'))
df_annotated = pd.read_csv(os.path.join(results_path.absolute(),"annotated_data.csv"))

# join df and df_annotated by model_name
df = df.merge(df_annotated, on='model_name')


df['f1'] = pd.to_numeric(df["f1"])
df['f1_lower'] = pd.to_numeric(df["f1_lower"])
df['f1_upper'] = pd.to_numeric(df["f1_upper"])

df.drop_duplicates(inplace=True)

df['dataset'] = df['dataset'].str.lower()

# add a color column to the dataframe with the color based on type column in rgba format with colors aqua for finetuned and orange for few shot and purple for zero shot and green for in context learning and faint grey for other
color_map = {'GPT2': 'rgba(230, 97, 0, 0.8)', 'GPT-Neo': 'rgba(75, 0, 146, 0.8)', 'zeroshot': 'rgba(75, 0, 146, 0.8)', 'icl': 'rgba(211, 95, 183, 0.8)', 'other': 'rgba(127, 127, 127, 0.5)'}
color_map_lines = {'finetuned': 'rgba(93, 58, 155, 1)', 'fewshot': 'rgba(255, 190, 106, 1)', 'zeroshot': 'rgba(26, 255, 26, 1)', 'icl': 'rgba(254, 254, 98, 1)', 'other': 'rgba(0, 0, 0, 1)'}

if average_all_datasets:
    ood_df = df[df['dataset'] != pandas_id_dataset.lower()]

    # average all f1 scores for datasets per model
    ood_df = ood_df.groupby(['model_name'])['f1', 'f1_lower', 'f1_upper'].mean()
    # zero out f1_lower and f1_upper
    ood_df['f1_lower'] = 0
    ood_df['f1_upper'] = 0
    ood_df = ood_df.reset_index()
else:
    ood_df = df.loc[df['dataset'] == pandas_dataset.lower()].drop(columns=['dataset'])
    
iid_df = df.loc[df['dataset'] == pandas_id_dataset.lower()].drop(columns=['dataset'])

ood_df = ood_df.drop(columns=['type', 'model_family', 'model_size', 'is_robust', 'is_adapter'], errors='ignore')
iid_df = iid_df.rename(columns={"f1": "iid_f1"})
iid_df = iid_df.rename(columns={"f1_lower": "iid_f1_lower"})
iid_df = iid_df.rename(columns={"f1_upper": "iid_f1_upper"})
ood_df = ood_df.rename(columns={"f1": "ood_f1"})
ood_df = ood_df.rename(columns={"f1_lower": "ood_f1_lower"})
ood_df = ood_df.rename(columns={"f1_upper": "ood_f1_upper"})

dataset_df = pd.concat([iid_df.set_index('model_name'), ood_df.set_index(
    'model_name')], axis=1, join='inner').reset_index()

dataset_df = dataset_df[dataset_df['type'] == 'fewshot']
    
def transform(f1, scaling=scaling):
    if type(f1) is list:
        f1 = np.array(f1)
    # divide by 100 to get the percentage
    f1 = f1 / 100
    if scaling == 'Linear':
        return f1
    elif scaling == 'Probit':
        return scipy.stats.norm.ppf(f1)
    elif scaling == 'Logit':
        return np.log(np.divide(f1, 1.0 - f1))

dataset_df['iid_f1_transformed'] = transform(dataset_df['iid_f1'], scaling=scaling)
dataset_df['ood_f1_transformed'] = transform(dataset_df['ood_f1'], scaling=scaling)

# Get x axis range
yrange = [0, 110]
xrange = [0, 110]

if scaling == 'Linear':
    dataset_df['iid_f1_lower'] = transform(dataset_df['iid_f1_lower'], scaling=scaling)
    dataset_df['iid_f1_upper'] = transform(dataset_df['iid_f1_upper'], scaling=scaling)
    dataset_df['ood_f1_lower'] = transform(dataset_df['ood_f1_lower'], scaling=scaling)
    dataset_df['ood_f1_upper'] = transform(dataset_df['ood_f1_upper'], scaling=scaling)
else:
    # remove bounds
    dataset_df.drop(columns=['iid_f1_lower', 'iid_f1_upper', 'ood_f1_lower', 'ood_f1_upper'], inplace=True)

# Create a dictionary of model_family: pretraining dataset
dataset_map = {"bert": "bookcorpus+wikipedia", "bart": "bookcorpus+wikipedia", "minilm": "bookcorpus+wikipedia",  "albert": "bookcorpus+wikipedia", "roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "gpt": "webtext", "bidaf":"none (Word2Vec or ELMO)", "spanbert": "bookcorpus+wikipedia", "t5": "C4", "adapter-roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "adapter-bert": "bookcorpus+wikipedia", "xlm-roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "gpt-neo": "pile", "splinter": "wikipedia+bookcorpus", "opt":"pile"}

# Add a pretrain_dataset column and iterate over the rows to add the pretrain_dataset
dataset_df['pretrain_dataset'] = dataset_df['model_family'].apply(lambda x: dataset_map[x])

hover_data = {"iid_f1": True, "ood_f1": True, "type": True, "model_family": True, "pretrain_dataset": True, "iid_f1_transformed": False, "ood_f1_transformed": False}

@st.cache()
def bootstrap_ci(df, n=1000):
    preds = []
    for _ in range(n):
        bootstrap_df = df.sample(frac=1, replace=True)
        # Add a trendline with confidence intervals using bootstrap
        z = np.polyfit(bootstrap_df['iid_f1_transformed'],
                        bootstrap_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(df['iid_f1_transformed'])
        preds.append(y_fit)

    return np.array(preds).mean(axis=0), preds
    
    
# sort by iid_f1_transformed
dataset_df.sort_values(by=['iid_f1_transformed'], inplace=True)

mean, preds = bootstrap_ci(dataset_df)
# Add trendline for few-shot models
z = np.polyfit(dataset_df['iid_f1_transformed'],
                dataset_df['ood_f1_transformed'], 1)
y_fit = np.poly1d(z)(dataset_df['iid_f1_transformed'])

# create a new column that is distance between ood_f1_transformed and y_fit
dataset_df['distance'] = (dataset_df['ood_f1_transformed'] - y_fit)*100

# create a df with models from the gpt and gpt-neo families
gpt_df = dataset_df[dataset_df['model_family'].isin(['gpt', 'gpt-neo'])]
# Add a num of shots column from the gpt and gpt-neo families with the same number of shots ie if -k-{num_shots} is the same
gpt_df['num_shots'] = gpt_df['model_name'].apply(lambda x: int(x.split('-k-')[-1].split('-')[0]))

# average the distance column for the gpt and gpt-neo families by num_shots and rename the column as {num_shots}_model
gpt_df = gpt_df.groupby(['num_shots', 'model_family']).mean().reset_index()

gpt_df['model_family'] = gpt_df['model_family'].apply(lambda x: 'GPT2' if x == 'gpt' else 'GPT-Neo' if x == 'gpt-neo' else x)

gpt_df['num_shots'] = gpt_df['num_shots'].apply(lambda x: np.log2(x))

fig = px.scatter(gpt_df, x="num_shots", y="distance", hover_name="num_shots", title="Average Effective Robustness of Fewshot GPT Models", labels=dict(iid_f1_transformed=f"F1 score performance on {pandas_id_dataset}", ood_f1_transformed=f"F1 score performance on {'all datasets' if average_all_datasets else pandas_dataset}"), opacity=0.8, symbol="model_family", color_discrete_map=color_map, color="model_family")

# add a trendline for gpt and gpt-neo separately
# gpt 2 trendline
z = np.polyfit(gpt_df[gpt_df['model_family'] == 'GPT2']['num_shots'], gpt_df[gpt_df['model_family'] == 'GPT2']['distance'], 1)
y_fit = np.poly1d(z)(gpt_df[gpt_df['model_family'] == 'GPT2']['num_shots'])
fig.add_scatter(x=gpt_df[gpt_df['model_family'] == 'GPT2']['num_shots'], y=y_fit, name='GPT2 Trendline', mode='lines', line=dict(color=color_map["GPT2"], width=6), showlegend=False)

#gpt-neo trendline
# apply log2 to all the num_shots
z = np.polyfit(gpt_df[gpt_df['model_family'] == 'GPT-Neo']['num_shots'], gpt_df[gpt_df['model_family'] == 'GPT-Neo']['distance'], 1)
y_fit = np.poly1d(z)(gpt_df[gpt_df['model_family'] == 'GPT-Neo']['num_shots'])
fig.add_scatter(x=gpt_df[gpt_df['model_family'] == 'GPT-Neo']['num_shots'], y=y_fit, name='GPT-Neo Trendline', mode='lines', line=dict(color=color_map["GPT-Neo"], width=6), showlegend=False)

# log scale x axis by np.log2
xticks = np.log2([16, 32, 64, 128, 256, 512, 1024])
fig.update_xaxes(tickvals=xticks, ticktext=[16, 32, 64, 128, 256, 512, 1024])


# Rename legend title
fig.update_layout(legend_title_text="")
# fig.update_layout(legend=dict(title_text="Model Family"))

# x axis title
fig.update_xaxes(title_text="Number of Shots")
# y axis title
fig.update_yaxes(title_text="Effective Robustness")

# put legend in the top right corner
fig.update_layout(legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99))
# set legend background to white
fig.update_layout(legend_bgcolor='white')

# Set plotly background color to transparent
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
# Set plotly grid lines to light grey
fig.update_xaxes(gridcolor='lightgrey', zerolinecolor='lightgrey', zerolinewidth=1)
fig.update_yaxes(gridcolor='lightgrey', zerolinecolor='lightgrey', zerolinewidth=1)

# set title only font size
fig.update_layout(font=dict(size=12))
fig.update_layout(title_font=dict(size=20))
fig.update_traces(marker=dict(size=12))
# make lines thicker
fig.update_traces(line=dict(width=5))
# Set plotly title to center
fig.update_layout(title_x=0.5)

# increase tick font size
fig.update_layout(xaxis_tickfont_size=14, yaxis_tickfont_size=14)

# have a bounding box around the plot
fig.update_layout(margin=dict(l=100, r=100, b=100, t=120))
# set background color to white
fig.update_layout(paper_bgcolor='rgba(255,255,255,0)')

# Add boarder
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(font=dict(color="black"))

st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Save Plot as Square"):
    fig.update_layout(height=1920, width=1920)
else:
    fig.update_layout(height=1080, width=1920)

fig.update_layout(font=dict(family="sans-serif"))
fig.update_layout(font=dict(size=72))
fig.update_layout(title_font=dict(size=76))
fig.update_traces(marker=dict(size=64))
fig.update_layout(xaxis_tickfont_size=54, yaxis_tickfont_size=54)
fig.update_layout(legend= {'itemsizing': 'trace'})

fig.update_layout(xaxis_tickfont_size=54, yaxis_tickfont_size=54)
# add padding between axes labels and tick labels
fig.update_xaxes(title_standoff = 50)
fig.update_yaxes(title_standoff = 50)

# set background color to transparent
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.write_image("plot.pdf")

with open("plot.pdf", "rb") as pdf:
    download_btn = st.download_button("Download Plot",
             data=pdf,
             file_name="plot.pdf",
             mime="pdf")
