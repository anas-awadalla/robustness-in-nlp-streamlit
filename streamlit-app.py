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

"""
The following datasets are available along with thier sizes:
'squad': 10570,
'squadshifts_nyt': 10065,
'squadshifts_reddit': 9803,
'squadshifts_new_wiki': 7938,
'squadshifts_amazon': 9885,
'RACE': 674, 
'DROP': 1503, 
'BioASQ': 1504,
'RelationExtraction': 2948,
'NewsQA': 4212,
'TriviaQA-web': 7785,
'SearchQA': 16980,
'HotpotQA': 5904,
'NaturalQuestionsShort': 12836
"""

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
#     'DuoRC.ParaphraseRC': 1501,
    'TriviaQA-web': 7785,
    'SearchQA': 16980,
    'HotpotQA': 5904,
    'NaturalQuestionsShort': 12836
}

dataset = st.sidebar.selectbox(
    'Dataset (Y-Axis)',
    list(dataset_to_size.keys()))

id_dataset = st.sidebar.selectbox(
    'Dataset (X-Axis)',
    list(dataset_to_size.keys()))

# if dataset == "DuoRC.ParaphraseRC":
#     pandas_dataset = "DuoRC(ParaphraseRC)"
# else:
pandas_dataset = dataset

# if id_dataset == "DuoRC.ParaphraseRC":
#     pandas_id_dataset = "DuoRC(ParaphraseRC)"
# else:
pandas_id_dataset = id_dataset

scaling = st.sidebar.selectbox("Graph Scaling", ['Linear', 'Logit', 'Probit'])
color_by_dataset = st.sidebar.checkbox("Color by Pretraining Dataset", value=False)

hide_zero_shot = st.sidebar.checkbox("Hide Zero Shot Models", value=False)
hide_few_shot = st.sidebar.checkbox("Hide Few Shot Models", value=False)
hide_icl = st.sidebar.checkbox("Hide In Context Learning Models", value=False)
hide_finetuned = st.sidebar.checkbox("Hide Finetuned Models", value=False)


results_path = Path(".") / "results"

df = pd.read_csv(os.path.join(results_path.absolute(),
                 'extractive_question_answering.csv'))
df.drop_duplicates(inplace=True)

df['dataset'] = df['dataset'].str.lower()

visible_models = st.sidebar.multiselect(label="Visible model families", options=list(df["model_family"].unique()), default=list(df["model_family"].unique()))
hidden_models = set(df["model_family"].unique()).difference(visible_models)

for hidden in hidden_models:
    df = df[df['model_family'] != hidden]

ood_df = df.loc[df['dataset'] == pandas_dataset.lower()].drop(columns=['dataset'])
iid_df = df.loc[df['dataset'] == pandas_id_dataset.lower()].drop(columns=['dataset'])

ood_df = ood_df.drop(columns=['type', 'model_family'])
iid_df = iid_df.rename(columns={"f1": "iid_f1"})
iid_df = iid_df.rename(columns={"f1_lower": "iid_f1_lower"})
iid_df = iid_df.rename(columns={"f1_upper": "iid_f1_upper"})
ood_df = ood_df.rename(columns={"f1": "ood_f1"})
ood_df = ood_df.rename(columns={"f1_lower": "ood_f1_lower"})
ood_df = ood_df.rename(columns={"f1_upper": "ood_f1_upper"})

dataset_df = pd.concat([iid_df.set_index('model_name'), ood_df.set_index(
    'model_name')], axis=1, join='inner').reset_index()

if hide_finetuned:
    dataset_df = dataset_df[dataset_df['type'] != 'finetuned']
    
if hide_zero_shot:
    dataset_df = dataset_df[dataset_df['type'] != 'zeroshot']
    
if hide_few_shot:
    dataset_df = dataset_df[dataset_df['type'] != 'fewshot']
    
if hide_icl:
    dataset_df = dataset_df[dataset_df['type'] != 'icl']
    
def transform(f1, scaling=scaling):
    if type(f1) is list:
        f1 = np.array(f1)
    if scaling == 'Linear':
        return (f1 / 100.0)
    elif scaling == 'Probit':
        return scipy.stats.norm.ppf(f1 / 100.0)
    elif scaling == 'Logit':
        return np.log(np.divide(f1 / 100.0, 1.0 - f1 / 100.0))
    
# Get x axis range
yrange = [0, 105]
xrange = [0, 105]

dataset_df['iid_f1_transformed'] = transform(dataset_df['iid_f1'], scaling=scaling)
dataset_df['ood_f1_transformed'] = transform(dataset_df['ood_f1'], scaling=scaling)

if scaling == 'Linear':
    dataset_df['iid_f1_lower'] = transform(dataset_df['iid_f1_lower'], scaling=scaling)
    dataset_df['iid_f1_upper'] = transform(dataset_df['iid_f1_upper'], scaling=scaling)
    dataset_df['ood_f1_lower'] = transform(dataset_df['ood_f1_lower'], scaling=scaling)
    dataset_df['ood_f1_upper'] = transform(dataset_df['ood_f1_upper'], scaling=scaling)
else:
    # remove bounds
    dataset_df.drop(columns=['iid_f1_lower', 'iid_f1_upper', 'ood_f1_lower', 'ood_f1_upper'], inplace=True)

# Create a dictionary of model_family: pretraining dataset
dataset_map = {"bert": "bookcorpus+wikipedia", "bart": "bookcorpus+wikipedia", "minilm": "bookcorpus+wikipedia",  "albert": "bookcorpus+wikipedia", "roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "gpt": "webtext", "bidaf":"none (Word2Vec or ELMO)", "spanbert": "bookcorpus+wikipedia", "t5": "C4", "adapter-roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "adapter-bert": "bookcorpus+wikipedia", "xlm-roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "gpt-neo": "pile", "splinter": "wikipedia+bookcorpus"}

# Add a pretrain_dataset column and iterate over the rows to add the pretrain_dataset
dataset_df['pretrain_dataset'] = dataset_df['model_family'].apply(lambda x: dataset_map[x])

hover_data = {"iid_f1": True, "ood_f1": True, "type": True, "model_family": True, "pretrain_dataset": True, "iid_f1_transformed": False, "ood_f1_transformed": False}

if scaling == 'Linear':
    fig = px.scatter(dataset_df, x="iid_f1_transformed", y="ood_f1_transformed", color="pretrain_dataset" if color_by_dataset else "model_family", hover_name="model_name",  hover_data=hover_data, error_x="iid_f1_upper", error_x_minus="iid_f1_lower", error_y="ood_f1_upper", error_y_minus="ood_f1_lower", title=f"Performance Comparison Between {pandas_id_dataset} and {pandas_dataset}", labels=dict(iid_f1_transformed=f"F1 Score Performance on {pandas_id_dataset}", ood_f1_transformed=f"F1 Score Performance on {pandas_dataset}"))

else:
    fig = px.scatter(dataset_df, x="iid_f1_transformed", y="ood_f1_transformed", color="pretrain_dataset" if color_by_dataset else "model_family", hover_name="model_name", hover_data=hover_data, title=f"Performance Comparison Between {pandas_id_dataset} and {pandas_dataset}", labels=dict(iid_f1_transformed=f"F1 Score Performance on {pandas_id_dataset}", ood_f1_transformed=f"F1 Score Performance on {pandas_dataset}"))


tick_loc_x = [round(z) for z in np.arange(xrange[0], xrange[1], 5)]
print(tick_loc_x)
print(transform(tick_loc_x, scaling=scaling))

# set x axis ticks
fig.update_xaxes(tickmode='array', tickvals=transform(tick_loc_x, scaling=scaling), ticktext=[str(z) for z in tick_loc_x])

tick_loc_y = [round(z) for z in np.arange(yrange[0], yrange[1], 5)]
print(tick_loc_y)
print(transform(tick_loc_y, scaling=scaling))
# set y axis ticks
fig.update_yaxes(tickmode='array', tickvals=transform(tick_loc_y, scaling=scaling), ticktext=[str(z) for z in tick_loc_y])

# turn off autoscale
fig.update_layout(autosize=False)

if not hide_finetuned:
    finetuned_df = dataset_df[dataset_df["type"] == "finetuned"]
    if len(finetuned_df) != 0:
        # Add trendline for finetuned models
        z = np.polyfit(finetuned_df['iid_f1_transformed'],
                        finetuned_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(finetuned_df['iid_f1_transformed'])

        line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(finetuned_df['ood_f1_transformed'] ,y_fit):0.3f}"
        fig.add_traces(go.Scatter(x=finetuned_df
                        ['iid_f1_transformed'], y=y_fit, name='Fine-Tuned Fit:'+line_equation, mode='lines'))

if not hide_zero_shot:
    zeroshot_df = dataset_df[dataset_df["type"] == "zeroshot"]
    if len(zeroshot_df) != 0:
        # Add trendline for zero-shot models
        z = np.polyfit(zeroshot_df['iid_f1_transformed'],
                        zeroshot_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(zeroshot_df['iid_f1_transformed'])

        line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(zeroshot_df['ood_f1_transformed'] ,y_fit):0.3f}"
        fig.add_traces(go.Scatter(x=zeroshot_df
                        ['iid_f1_transformed'], y=y_fit, name='Zero-Shot Fit:'+line_equation, mode='lines'))

if not hide_few_shot:
    fewshot_df = dataset_df[dataset_df["type"] == "fewshot"]
    if len(fewshot_df) != 0:
        # Add trendline for few-shot models
        z = np.polyfit(fewshot_df['iid_f1_transformed'],
                        fewshot_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(fewshot_df['iid_f1_transformed'])

        line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(fewshot_df['ood_f1_transformed'] ,y_fit):0.3f}"
        fig.add_traces(go.Scatter(x=fewshot_df
                        ['iid_f1_transformed'], y=y_fit, name='Few-Shot Fit:'+line_equation, mode='lines'))

if not hide_icl:
    icl_df = dataset_df[dataset_df["type"] == "icl"]
    if len(icl_df) != 0:
        # Add trendline for icl models
        z = np.polyfit(icl_df['iid_f1_transformed'],
                        icl_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(icl_df['iid_f1_transformed'])

        line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(icl_df['ood_f1_transformed'] ,y_fit):0.3f}"
        fig.add_traces(go.Scatter(x=icl_df
                        ['iid_f1_transformed'], y=y_fit, name='ICL Fit:'+line_equation, mode='lines'))

# Plot y=x line using tick values
fig.add_trace(go.Line(x=transform(tick_loc_y), y=transform(tick_loc_y), mode='lines', name='y=x', line_dash="dash", line_color="blue",hoverinfo='none'))

st.plotly_chart(fig, use_container_width=True)

# remove transformed f1 cols
dataset_df.drop(["iid_f1_transformed", "ood_f1_transformed"], axis=1, inplace=True)

dataset_df = dataset_df.rename(columns={"iid_f1": "id_f1"})
dataset_df = dataset_df.rename(columns={"iid_f1_upper": "id_f1_upper"})
dataset_df = dataset_df.rename(columns={"iid_f1_lower": "id_f1_lower"})


st.dataframe(dataset_df)
