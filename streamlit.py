from collections import defaultdict
import json
import numpy as np
import pandas as pd
import scipy
import streamlit as st
import plotly.express as px
from sklearn.utils import resample
from transformers.data.metrics.squad_metrics import compute_f1
import glob
import os
import plotly.graph_objects as go


dataset = st.sidebar.selectbox(
    'Dataset',
    ('squadshifts_nyt', 'squadshifts_reddit', 'squadshifts_new_wiki',
     'squadshifts_amazon', 'RACE', 'DROP', 'TextbookQA', 'BioASQ', 'RelationExtraction')
)

df = pd.read_csv('results/extractive_question_answering.csv')


"""
Select the dataset to visualize from the sidebar. For bootstrap use n to set the number of samples at each iteration. Use N to set the number of trials.
"""

n_samples = int(st.number_input("n For Bootstrap", min_value=1, value=100))
num_iterations = int(st.number_input(
    "N For Bootstrap", min_value=1, value=100))

selected_file = None
def bootstrap(model, dataset_name, n_samples, num_iterations):
    for file in glob.glob(f'../results/*.json'):
        file_name = os.path.basename(file).split('.')[0].lower()
        if file_name.find(model.split('/')[-1].lower()) != -1 and file_name.find(dataset_name.lower()) != -1:
            selected_file = file

    with open(selected_file) as f:
        data = json.load(f)

    f1_scores = []

    for id in data.keys():
        predicted_text = data[id]['prediction']['prediction_text']
        actual_text = data[id]['reference']['answers']['text']
        max_f1 = float("-inf")
        for possible_answer in actual_text:
            max_f1 = max(max_f1, 100.0 * compute_f1(possible_answer, predicted_text))
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
        model, dataset, n_samples, num_iterations)

ood_df = df.loc[df['dataset_name'] == dataset].drop(columns=['dataset_name'])
iid_df = df.loc[df['dataset_name'] == "squad"].drop(columns=['dataset_name'])

ood_df['ood_bootstrap_f1'] = ood_df['model_name'].apply(
    lambda x: ood_bootstrap_f1[x][0])
ood_df['e_minus'] = ood_df['model_name'].apply(
    lambda x: abs(ood_bootstrap_f1[x][1][0] - ood_bootstrap_f1[x][0]))
ood_df['e_plus'] = ood_df['model_name'].apply(
    lambda x: abs(ood_bootstrap_f1[x][1][1] - ood_bootstrap_f1[x][0]))

iid_df = iid_df.rename(columns={"f1": "iid_f1"})
ood_df = ood_df.drop(columns=['zero_shot'])

dataset_df = pd.concat([iid_df.set_index('model_name'), ood_df.set_index(
    'model_name')], axis=1, join='inner').reset_index()
dataset_df['zero_shot'] = dataset_df['zero_shot'].astype('bool')

fig = px.scatter(dataset_df, x="iid_f1", y="ood_bootstrap_f1", color="zero_shot",
                 hover_data=["model_name"], error_y="e_plus", error_y_minus="e_minus",
                 trendline="ols", title=f"Performance Comparison Between squad and {dataset}")

st.plotly_chart(fig, use_container_width=True)
