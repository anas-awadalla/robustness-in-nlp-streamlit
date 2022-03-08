from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import streamlit as st
import plotly.express as px
from sklearn.utils import resample
from transformers.data.metrics.squad_metrics import compute_f1
import os
import plotly.graph_objects as go


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
'RelationExtraction': 2948
"""
dataset_to_size = {
    'squad': 10570,
    'squadshifts_nyt': 10065,
    'squadshifts_reddit': 9803,
    'squadshifts_new_wiki': 7938,
    'squadshifts_amazon': 9885,
    'RACE': 674, 
    'DROP': 1503, 
    # 'TextbookQA': 1503,
    'BioASQ': 1504,
    'RelationExtraction': 2948
}

dataset = st.sidebar.selectbox(
    'Dataset (Y-Axis)',
    ('squadshifts_nyt', 'squad', 'squadshifts_reddit', 'squadshifts_new_wiki',
     'squadshifts_amazon', 'RACE', 'DROP', 'BioASQ', 'RelationExtraction'))

id_dataset = st.sidebar.selectbox(
    'Dataset (X-Axis)',
    ('squad', 'squadshifts_nyt', 'squadshifts_reddit', 'squadshifts_new_wiki',
     'squadshifts_amazon', 'RACE', 'DROP', 'BioASQ', 'RelationExtraction'))

hide_zero_shot = st.checkbox("Hide Zero Shot Models", value=False)
hide_few_shot = st.checkbox("Hide Few Shot Models", value=False)
hide_icl = st.checkbox("Hide In Context Learning Models", value=False)
hide_finetuned = st.checkbox("Hide Finetuned Models", value=False)

n_samples_ood = dataset_to_size[dataset]
num_iterations_ood = 1000
n_samples_iid = dataset_to_size[id_dataset]
num_iterations_iid = 1000

results_path = Path(".") / "results"

df = pd.read_csv(os.path.join(results_path.absolute(),
                 'extractive_question_answering.csv'))


@st.experimental_memo(show_spinner=True)
def bootstrap(model, dataset_name, n_samples, num_iterations):
    for file in results_path.glob(f'*.json'):
        file_name = os.path.basename(file).split('.')[0].lower()
        if dataset_name.find("_") == -1:
            data_file_section = file_name.split('_')[-1]
        else:
            data_file_section = file_name.split('_', 1)[-1]
        if file_name.find(model.split('/')[-1].lower()) != -1 and data_file_section.find(dataset_name.lower()) != -1:
            selected_file = file
    
    with open(selected_file) as f:
        data = json.load(f)

    f1_scores = []

    for id in data.keys():
        predicted_text = data[id]['prediction']['prediction_text']
        actual_text = data[id]['reference']['answers']['text']
        max_f1 = float("-inf")
        for possible_answer in actual_text:
            max_f1 = max(max_f1, 100.0 *
                         compute_f1(possible_answer, predicted_text))
        f1_scores.append(max_f1)

    mean_list = []
    for _ in range(num_iterations):
        samples = resample(f1_scores, n_samples=n_samples, replace=True)
        sample_mean = np.mean(samples)
        mean_list.append(sample_mean)
    overall_mean = np.mean(mean_list)
    conf_interval = np.percentile(mean_list, [2.5, 97.5])
    return overall_mean, conf_interval


ood_bootstrap_f1 = defaultdict()
for model in df['model_name'].unique():
    ood_bootstrap_f1[model] = bootstrap(
        model, dataset, n_samples_ood, num_iterations_ood)

iid_bootstrap_f1 = defaultdict()
for model in df['model_name'].unique():
    iid_bootstrap_f1[model] = bootstrap(
        model, id_dataset, n_samples_iid, num_iterations_iid)

ood_df = df.loc[df['dataset_name'] == dataset].drop(columns=['dataset_name'])
iid_df = df.loc[df['dataset_name'] == id_dataset].drop(columns=['dataset_name'])

ood_df['ood_bootstrap_f1'] = ood_df['model_name'].apply(
    lambda x: ood_bootstrap_f1[x][0])
ood_df['e_minus_ood'] = ood_df['model_name'].apply(
    lambda x: abs(ood_bootstrap_f1[x][1][0] - ood_bootstrap_f1[x][0]))
ood_df['e_plus_ood'] = ood_df['model_name'].apply(
    lambda x: abs(ood_bootstrap_f1[x][1][1] - ood_bootstrap_f1[x][0]))

iid_df['iid_bootstrap_f1'] = iid_df['model_name'].apply(
    lambda x: iid_bootstrap_f1[x][0])
iid_df['e_minus_iid'] = iid_df['model_name'].apply(
    lambda x: abs(iid_bootstrap_f1[x][1][0] - iid_bootstrap_f1[x][0]))
iid_df['e_plus_iid'] = iid_df['model_name'].apply(
    lambda x: abs(iid_bootstrap_f1[x][1][1] - iid_bootstrap_f1[x][0]))

ood_df = ood_df.drop(columns=['k_shot', 'model_family', 'exact_match'])
iid_df = iid_df.rename(columns={"f1": "iid_f1"})
ood_df = ood_df.rename(columns={"f1": "ood_f1"})

dataset_df = pd.concat([iid_df.set_index('model_name'), ood_df.set_index(
    'model_name')], axis=1, join='inner').reset_index()
#dataset_df['zero_shot'] = dataset_df['zero_shot'].astype('bool')
#dataset_df = dataset_df.rename(columns={"zero_shot": "k_shot"})

fig = px.scatter(dataset_df, x="iid_f1", y="ood_f1", color="model_family",
                 hover_data=["model_name", "k_shot"], error_x="e_plus_iid", error_x_minus="e_minus_iid",
                 error_y="e_plus_ood", error_y_minus="e_minus_ood", title=f"Performance Comparison Between {id_dataset} and {dataset}",
                 labels=dict(iid_f1=f"F1 Score Performance on {id_dataset}", ood_f1=f"F1 Score Performance on {dataset}"))

if not hide_finetuned:
    # Add trendline for finetuned models
    z = np.polyfit(dataset_df[dataset_df['type'] == 'finetuned']['iid_f1'],
                   dataset_df[dataset_df['type'] == 'finetuned']['ood_f1'], 1)
    y_fit = np.poly1d(z)(dataset_df[dataset_df['type'] == 'finetuned']['iid_f1'])

    line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(dataset_df[dataset_df['type'] == 'finetuned']['ood_f1'] ,y_fit):0.3f}"
    fig.add_traces(go.Scatter(x=dataset_df[dataset_df['type'] == 'finetuned']
                   ['iid_f1'], y=y_fit, name='Fine-Tuned Fit:'+line_equation, mode='lines'))

if not hide_zero_shot:
    # Add trendline for zero-shot models
    z = np.polyfit(dataset_df[dataset_df['type'] == 'zero_shot']['iid_f1'],
                   dataset_df[dataset_df['type'] == 'zero_shot']['ood_f1'], 1)
    y_fit = np.poly1d(z)(dataset_df[dataset_df['type'] == 'zero_shot']['iid_f1'])

    line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(dataset_df[dataset_df['type'] == 'zero_shot']['ood_f1'] ,y_fit):0.3f}"
    fig.add_traces(go.Scatter(x=dataset_df[dataset_df['type'] == 'zero_shot']
                   ['iid_f1'], y=y_fit, name='Zero-Shot Fit:'+line_equation, mode='lines'))

if not hide_few_shot:
    # Add trendline for few-shot models
    z = np.polyfit(dataset_df[dataset_df['type'] == 'few_shot']['iid_f1'],
                   dataset_df[dataset_df['type'] == 'few_shot']['ood_f1'], 1)
    y_fit = np.poly1d(z)(dataset_df[dataset_df['type'] == 'few_shot']['iid_f1'])

    line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(dataset_df[dataset_df['type'] == 'few_shot']['ood_f1'] ,y_fit):0.3f}"
    fig.add_traces(go.Scatter(x=dataset_df[dataset_df['type'] == 'few_shot']
                   ['iid_f1'], y=y_fit, name='Few-Shot Fit:'+line_equation, mode='lines'))

if not hide_icl:
    # Add trendline for icl models
    z = np.polyfit(dataset_df[dataset_df['type'] == 'icl']['iid_f1'],
                   dataset_df[dataset_df['type'] == 'icl']['ood_f1'], 1)
    y_fit = np.poly1d(z)(dataset_df[dataset_df['type'] == 'icl']['iid_f1'])

    line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(dataset_df[dataset_df['type'] == 'icl']['ood_f1'] ,y_fit):0.3f}"
    fig.add_traces(go.Scatter(x=dataset_df[dataset_df['type'] == 'icl']
                   ['iid_f1'], y=y_fit, name='ICL Fit:'+line_equation, mode='lines'))

fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=100,
                y1=100,
                line=dict(color='Blue',),
                line_dash="dash",
                xref='x',
                yref='y')

dataset_df = dataset_df.rename(columns={"iid_f1": "id_f1"})
dataset_df = dataset_df.drop(columns=["iid_bootstrap_f1", "ood_bootstrap_f1"])
dataset_df = dataset_df.rename(columns={"e_minus_iid": "e_minus_id"})
dataset_df = dataset_df.rename(columns={"e_plus_iid": "e_plus_id"})


st.plotly_chart(fig, use_container_width=True)
st.dataframe(dataset_df)
