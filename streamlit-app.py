import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import streamlit as st
from sklearn.metrics import r2_score

from plotter import Plotter, Scaling, config
from utils import *

st.set_page_config(
    page_title="QA Robustness"
)

# st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

# with st.expander("See explanation"):
#     st.write("""
#         The chart above shows some numbers I picked for you.
#         I rolled actual dice for these, so they're *guaranteed* to
#         be random.
#     """)
#     st.image("https://static.streamlit.io/examples/dice.jpg")

testsets = ["squad",
            "squadshifts_nyt",
            "squadshifts_reddit",
            "squadshifts_new_wiki",
            "squadshifts_amazon",
            "RACE", "DROP", "TextbookQA",
            "BioASQ", "RelationExtraction",
            "NewsQA", "DuoRC.ParaphraseRC",
            "TriviaQA-web", "SearchQA",
            "HotpotQA", "NaturalQuestionsShort"]

average_all_datasets = st.sidebar.checkbox("Average All Datasets", value=False)

dataset = st.sidebar.selectbox(
    'Dataset (Y-Axis)', testsets, disabled=average_all_datasets)

id_dataset = st.sidebar.selectbox(
    'Dataset (X-Axis)',
    testsets)

scaling = st.sidebar.selectbox(
    "Graph Scaling", [Scaling.LINEAR, Scaling.PROBIT, Scaling.LOGIT])
use_plotly = st.sidebar.selectbox(
    "Plotting Framework", ["Plotly", "Matplotlib"], disabled=False) == "Plotly"

show_only_baseline = False #st.sidebar.checkbox("Show Only Baseline", value=False, disabled=True)
group_by_finetuned = st.sidebar.checkbox("Group By Finetuned", value=True, disabled=True)

results_path = Path(".") / "results"

df = pd.read_csv(os.path.join(results_path.absolute(),
                 'extractive_question_answering_final_avg.csv'))

larger_family_to_model = {"Encoder Model": ["bert", "albert", "roberta", "spanbert", "splinter", "distilbert"], "Decoder Model": ["gpt", "opt", "gpt-neo", "gpt-j"], "Encoder-Decoder Model": ["bart", "t5"]}

inverse_larger_family_to_model = {}

for key, value in larger_family_to_model.items():
    for v in value:
        inverse_larger_family_to_model[v] = key

df_annotated = pd.read_csv(os.path.join(
    results_path.absolute(), "annotated_data.csv"))

# join df and df_annotated by model_name
df = df.merge(df_annotated, on='model_name')

df = df[~df['model_name'].str.contains('wise')]

df['type'] = df['type'].apply(lambda x: x.replace('prompt finetuned', 'Prompt Fine-tuned (PT)').replace('finetuned', 'Fine-tuned (FT)' if group_by_finetuned else 'Baseline (Fine-tuned)').replace(
    'zeroshot', 'Zero-shot').replace('fewshot', 'Few-shot').replace('icl', 'In-Context Learning'))

# if model_family contains gpt and type is Few-shot then rename type to Few-shot Seq2Seq
for index, row in df.iterrows():
    if ((('gpt' in row['model_family'] or 'opt' in row['model_family']) and 'span-head' not in row['model_name']) or ('bart' in row['model_family'] and 'mask-fill' in row['model_name']) or ('t5' in row['model_family'] and 'mask-fill' in row['model_name'])) and row['type'] == 'Few-shot':
        df.at[index, 'type'] = 'Few-shot PT (Alligned)'
    elif (('bart' in row['model_family'] and 'seq2seq' in row['model_name']) or ('t5' in row['model_family'] and 'seq2seq' in row['model_name'])) and row['type'] == 'Few-shot':
        df.at[index, 'type'] = 'Few-shot PT (Unalligned)'
    elif row['type'] == 'Few-shot':
        df.at[index, 'type'] = 'Few-shot FT'
    elif row['type'] == 'Baseline (Fine-tuned)' and row["is_adapter"]:
        df.at[index, 'type'] = 'Parameter Efficient'
        # if 'prefix' in row['model_name']:
        #     df.at[index, 'type'] = 'Prefix tuning'
        # elif 'houlsby' in row['model_name'] or 'pfeiffer' in row['model_name']:
        #     df.at[index, 'type'] = 'Adapter tuning'
        # elif 'lora' in row['model_name']:
        #     df.at[index, 'type'] = 'LoRA tuning'
        # else:
        #     print("Unknown adapter type - ", row['model_name'])
    elif row['type'] == 'Baseline (Fine-tuned)' and row["is_robust"]:
        df.at[index, 'type'] = 'Robustness Enhanced'

if st.sidebar.checkbox("Hide Zero Shot Models", value=False):
    df = df[df['type'] != 'Zero-shot']
if st.sidebar.checkbox("Hide Few Shot Models", value=False):
    df = df[df['type'] != 'Few-shot FT']
    df = df[df['type'] != 'Few-shot PT (Alligned)']
    df = df[df['type'] != 'Few-shot PT (Unalligned)']
if st.sidebar.checkbox("Hide In Context Learning Models", value=False):
    df = df[df['type'] != 'In-Context Learning']
if st.sidebar.checkbox("Hide Parameter Efficient Models", value=False, disabled=True):
    df = df[df['type'] != 'Parameter Efficient']
if st.sidebar.checkbox("Hide Robustness Enhanced Models", value=False, disabled=True):
    df = df[df['type'] != 'Robustness Enhanced']

if show_only_baseline:
    df = df[df['type'] != 'Robustness Enhanced']
    df = df[df['type'] != 'Parameter Efficient']

df['f1'] = pd.to_numeric(df["f1"])
df['f1_lower'] = pd.to_numeric(df["f1_lower"])
df['f1_upper'] = pd.to_numeric(df["f1_upper"])

df.drop_duplicates(inplace=True)

df['dataset'] = df['dataset'].str.lower()

visible_models = st.sidebar.multiselect(label="Visible model families", options=list(
    df["model_family"].unique()), default=list(df["model_family"].unique()))
hidden_models = set(df["model_family"].unique()).difference(visible_models)

for hidden in hidden_models:
    df = df[df['model_family'] != hidden]

if average_all_datasets:
    ood_df = df[df['dataset'] != dataset.lower()]

    # average all f1 scores for datasets per model
    ood_df = ood_df.groupby(['model_name'])[
        'f1', 'f1_lower', 'f1_upper'].mean()
    # zero out f1_lower and f1_upper
    ood_df['f1_lower'] = 0
    ood_df['f1_upper'] = 0
    ood_df = ood_df.reset_index()
else:
    ood_df = df.loc[df['dataset'] == dataset.lower()].drop(columns=[
        'dataset'])

iid_df = df.loc[df['dataset'] == id_dataset.lower()].drop(columns=[
    'dataset'])

ood_df = ood_df.drop(columns=['type', 'model_family',
                     'model_size', 'is_robust', 'is_adapter'], errors='ignore')
iid_df = iid_df.rename(
    columns={"f1": "iid_f1", "f1_lower": "iid_f1_lower", "f1_upper": "iid_f1_upper"})
ood_df = ood_df.rename(
    columns={"f1": "ood_f1", "f1_lower": "ood_f1_lower", "f1_upper": "ood_f1_upper"})

dataset_df = pd.concat([iid_df.set_index('model_name'), ood_df.set_index(
    'model_name')], axis=1, join='inner').reset_index()

color_map = {'Fine-tuned (FT)' if group_by_finetuned else 'Baseline (Fine-tuned)': 'rgba(230, 97, 0, 0.4)',
             "Prompt Fine-tuned (PT)": 'rgba(246,190,0, 0.5)',
             'Few-shot FT': 'rgba(26, 133, 255, 0.4)',
             'Few-shot PT (Alligned)': 'rgba(64, 176, 166, 0.4)',
             'Few-shot PT (Unalligned)': 'rgba(0, 0, 0, 0.4)',
             'Zero-shot': 'rgba(75, 0, 146, 0.4)',
             'In-Context Learning': 'rgba(212, 17, 89, 0.4)',
             'Robustness Enhanced': 'rgba(0, 0, 0, 0.4)',
             'Parameter Efficient': 'rgba(128,128,0, 0.4)'}

color_map_mpl = {'Fine-tuned (FT)' if group_by_finetuned else 'Baseline (Fine-tuned)': 'darkorange',
                 'Prompt Fine-tuned (PT)': 'gold',
                 'Few-shot FT': 'royalblue',
                 'Few-shot PT (Alligned)': 'teal',
                 'Few-shot PT (Unalligned)': 'black',
                 'Zero-shot': 'darkviolet',
                 'In-Context Learning': 'crimson',
                 'Robustness Enhanced': 'black',
                 'Parameter Efficient': 'darkgreen'}


symbol_map = {'Fine-tuned (FT)' if group_by_finetuned else 'Baseline (Fine-tuned)': "o",
              "Prompt Fine-tuned (PT)": "^",
              "Zero-shot": "D",
              "In-Context Learning": "P",
              "Few-shot PT (Alligned)": "X",
              "Few-shot PT (Unalligned)": "X",
              "Few-shot FT": "s",
              "Robustness Enhanced": "*",
              "Parameter Efficient": "d"}

# symbol_map = {'Encoder Model': "o",
#               "Decoder Model": "D",
#                "Encoder-Decoder Model": "P",
#               "Few-shot Seq2Seq": "X",
#               "Few-shot FT": "s",
#               "Robustness Enhanced": "*",
#               "Parameter Efficient": "d"}
# color_map_mpl = {'Encoder Model': 'darkorange',
#                     'Decoder Model': 'royalblue',
#                     'Encoder-Decoder Model': 'darkviolet',
#                     'Few-shot Seq2Seq': 'teal',
#                     'Few-shot FT': 'teal',
#                     'Robustness Enhanced': 'black',
#                     'Parameter Efficient': 'darkgreen'}
# color_map = {'Encoder Model',
#                     'Decoder Model': 'rgba(26, 133, 255, 0.4)',
#                     'Encoder-Decoder Model': 'rgba(212, 17, 89, 0.4)',
#                     'Few-shot Seq2Seq': 'rgba(64, 176, 166, 0.4)',
#                     'Few-shot FT': 'rgba(26, 133, 255, 0.4)',
#                     'Robustness Enhanced': 'rgba(0, 0, 0, 0.4)',
#                     'Parameter Efficient': 'rgba(128,128,0, 0.4)'}

in_distribution_label = 'SQuAD' if id_dataset == 'squad' else id_dataset

if average_all_datasets:
    out_distribution_label = 'average of all datasets'
else:
    out_distribution_label = 'SQuAD' if dataset == 'squad' else dataset

# map model_name to larger family
dataset_df['model_architecture'] = dataset_df['model_family'].apply(lambda x: inverse_larger_family_to_model[x])
dataset_df.rename(columns={'model_name': 'Model', "iid_f1": "ID F1", "ood_f1": "OOD F1"}, inplace=True)

plot_config = config(title=f"{in_distribution_label} → {out_distribution_label} shift", x_label="ID F1", group_by="type",
                     y_label="OOD F1", scaling_type=scaling, plotly=use_plotly,
                     color_map=color_map if use_plotly else color_map_mpl, show_legend=True,
                     plot_x_label=f"F1 score on {in_distribution_label}",
                     plot_y_label=f"F1 score on {out_distribution_label}",
                     symbol_map=symbol_map)

fig = Plotter.plot(dataset_df, plot_config)

if plot_config.plotly:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.pyplot(fig, use_container_width=True)
    # save figure to file

# file_name = st.text_input("Save figure as", plot_config.title)
# if st.button("Save Plot") and not plot_config.plotly:
#     fig.savefig(f"images/{file_name}.pdf")
