from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
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

results_path = Path(".") / "results"

df = pd.read_csv(os.path.join(results_path.absolute(),'extractive_question_answering.csv'))

# model_name = st.sidebar.selectbox(
#     'Model',
#     df['model_name'].unique()
# )


"""
Select the dataset to visualize from the sidebar. For bootstrap use n to set the number of samples at each iteration. Use N to set the number of trials.
"""

n_samples = int(st.number_input("n For Bootstrap", min_value=1, value=100))
num_iterations = int(st.number_input(
    "N For Bootstrap", min_value=1, value=100))

def bootstrap(model, dataset_name, n_samples, num_iterations):
    for file in results_path.glob(f'*.json'):
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

# def adjust_plot(fig, xrange, yrange, scaling, tick_freq=0.05):
#     if type(tick_freq) is tuple:
#         xtick_freq, ytick_freq = tick_freq[0], tick_freq[1]
#     else:
#         xtick_freq, ytick_freq = tick_freq, tick_freq

#     if scaling == 'probit':
#         h = scipy.stats.norm.ppf
#     elif scaling == 'logit':
#         h = lambda p: np.log(p / (1 - p))
#     else:
#         h = lambda p: p

#     def transform(z):
#         return [h(p) for p in z]

#     tick_loc_x = [round(z, 2) for z in np.arange(xrange[0], xrange[1], xtick_freq)]
    
#     fig.update_layout(
#         xaxis = dict(
#             tickmode = 'array',
#             tickvals = transform(tick_loc_x),
#             ticktext = [str(round(loc * 100)) for loc in tick_loc_x]
#         )
#     )

#     tick_loc_y = [round(z, 2) for z in np.arange(yrange[0], yrange[1], ytick_freq)]
    
#     fig.update_layout(
#         yaxis = dict(
#             tickmode = 'array',
#             tickvals = transform(tick_loc_y),
#             ticktext = [str(round(loc * 100)) for loc in tick_loc_y]
#         )
#     )

#     fig.update_layout(yaxis_range=(h(yrange[0]), h(yrange[1])))
#     fig.update_layout(xaxis_range=(h(xrange[0]), h(xrange[1])))

#     return transform, fig

# y_range = (0.95 * min(iid_df["iid_f1"]), 1.05 * max(iid_df["iid_f1"]))
# x_range = (0.95 * min(ood_df["ood_bootstrap_f1"]), 1.05 * max(ood_df["ood_bootstrap_f1"]))
# print(x_range)
# print(y_range)

# transform, fig = adjust_plot(fig, x_range, y_range, scaling='logit')

dataset_df = pd.concat([iid_df.set_index('model_name'), ood_df.set_index(
    'model_name')], axis=1, join='inner').reset_index()
dataset_df['zero_shot'] = dataset_df['zero_shot'].astype('bool')

fig = px.scatter(dataset_df, x="iid_f1", y="ood_bootstrap_f1", color="zero_shot",
                 hover_data=["model_name"], error_y="e_plus", error_y_minus="e_minus",
                 trendline="ols", title=f"Performance Comparison Between squad and {dataset}")

st.plotly_chart(fig, use_container_width=True)

# else:
#     """
#     Explore how the model performs on a specific dataset. Use the slider to select a specific datapoint.
#     """

#     # load json file


#     selected_file = None
#     for file in glob.glob(f'/home/mitchnw/ssd/anas/robustness-in-nlp/src/**/*.json'):
#         file_name = os.path.basename(file).split('.')[0].lower()
#         print(file_name)
#         if file_name.find(model_name.split('/')[-1].lower()) != -1 and file_name.find(dataset.lower()) != -1:
#             selected_file = file

#     if selected_file is None:
#         st.error('Could not find model results for dataset {}'.format(dataset))
#     else:
#         with open(selected_file) as f:
#             data = json.load(f)

#         m = defaultdict()

#         # create dataframe
#         for new_idx, id in enumerate(data.keys()):
#             m[new_idx] = {'input': data[id]['input'], 'predicted answer': data[id]['prediction']['prediction_text'], 'groundtruth answers': data[id]['reference']['answers']['text']}

#         # df = pd.DataFrame.from_dict(m, orient='index').reset_index(drop=True, inplace=False)

#         idx = st.slider('Datapoint Index', 0, len(m))

#         st.json(m[idx])
