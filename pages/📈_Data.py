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
    page_title="Question Answering Robustness",
    layout="centered",
)

st.markdown("## Explore our evaluations data!")
st.markdown("""By default we show the average performance for each model across all datasets and apply logit scaling to the axes. 
            You can change these settings in the sidebar.""")
st.markdown("""Explore performance on specific datasets by unchecking the 'Average across datasets' 
            box and selecting the datasets you want to see in the 'Y-axis' dropdown.""")

# set font for matplotlib as times new roman
# plt.rcParams["font.family"] = "Times New Roman"

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

average_all_datasets = st.sidebar.checkbox("Average across datasets", value=True)

dataset = st.sidebar.selectbox(
    'Dataset (Y-Axis)', testsets, disabled=average_all_datasets)

id_dataset = st.sidebar.selectbox(
    'Dataset (X-Axis)',
    testsets)

scaling = st.sidebar.selectbox(
    "Graph Scaling", [Scaling.LINEAR, Scaling.PROBIT, Scaling.LOGIT], index=2)
use_plotly = st.sidebar.selectbox(
    "Plotting Framework", ["Plotly", "Matplotlib"], disabled=False) == "Plotly"

show_only_baseline = True #st.sidebar.checkbox("Show Only Baseline", value=False, disabled=True)

results_path = Path(".") / "results"

df = pd.read_csv(os.path.join(results_path.absolute(),
                 'extractive_question_answering_final_avg.csv'))

larger_family_to_model = {"Encoder-only Models": ["bert", "albert", "roberta", "spanbert", "splinter", "distilbert"], "Decoder-only Models": ["gpt", "opt", "gpt-neo", "gpt-j"], "Encoder-Decoder Models": ["bart", "t5"]}

inverse_larger_family_to_model = {}

for key, value in larger_family_to_model.items():
    for v in value:
        inverse_larger_family_to_model[v] = key

df_annotated = pd.read_csv(os.path.join(
    results_path.absolute(), "annotated_data.csv"))

# join df and df_annotated by model_name
df = df.merge(df_annotated, on='model_name')

df = df[~df['model_name'].str.contains('wise')]

df['type'] = df['type'].apply(lambda x: x.replace('prompt finetuned', 'Prompt Fine-tuned').replace('finetuned', 'Fine-tuned' ).replace(
    'zeroshot', 'Zero-shot').replace('fewshot', 'Few-shot').replace('icl', 'In-Context Learning'))

for index, row in df.iterrows():
    if (('bart' in row['model_family'] and 'seq2seq' in row['model_name']) or ('t5' in row['model_family'] and 'seq2seq' in row['model_name'])) and row['type'] == 'Few-shot':
        df.drop(index, inplace=True)

# if model_family contains gpt and type is Few-shot then rename type to Few-shot Seq2Seq
for index, row in df.iterrows():
    if ((('gpt' in row['model_family'] or 'opt' in row['model_family']) and 'span-head' not in row['model_name']) or ('bart' in row['model_family'] and 'mask-fill' in row['model_name']) or ('t5' in row['model_family'] and 'mask-fill' in row['model_name'])) and row['type'] == 'Few-shot':
        df.at[index, 'type'] = 'Few-shot Prompt Fine-tuned'
    elif row['type'] == 'Few-shot':
        df.at[index, 'type'] = 'Few-shot Fine-tuned'
    # elif row['type'] == 'Fine-tuned' and row["is_adapter"]:
    #     df.at[index, 'type'] = 'Parameter Efficient Fine-tuned'
    # elif row['type'] == 'Prompt Fine-tuned' and row["is_adapter"]:
    #     df.at[index, 'type'] = 'Parameter Efficient Prompt Fine-tuned'
        # if 'prefix' in row['model_name']:
        #     df.at[index, 'type'] = 'Prefix tuning'
        # elif 'houlsby' in row['model_name'] or 'pfeiffer' in row['model_name']:
        #     df.at[index, 'type'] = 'Adapter tuning'
        # elif 'lora' in row['model_name']:
        #     df.at[index, 'type'] = 'LoRA tuning'
        # else:
        #     print("Unknown adapter type - ", row['model_name'])
    # elif row['type'] == 'Fine-tuned' and row["is_robust"]:
    #     df.at[index, 'type'] = 'Robustness Enhanced Fine-tuned'


# if st.sidebar.checkbox("Hide Zero Shot Models", value=False):
#     df = df[df['type'] != 'Zero-shot']
# if st.sidebar.checkbox("Hide Few Shot Models", value=False):
#     df = df[df['type'] != 'Few-shot Fine-tuned']
#     df = df[df['type'] != 'Few-shot Prompt Fine-tuned']
# if st.sidebar.checkbox("Hide In Context Learning Models", value=False):
#     df = df[df['type'] != 'In-Context Learning']
# if st.sidebar.checkbox("Hide Parameter Efficient Models", value=False, disabled=False):
#     df = df[df['is_adapter'] != True]
# if st.sidebar.checkbox("Hide Robustness Enhanced Models", value=False, disabled=False):
#     df = df[df['is_robust'] != True]
# if st.sidebar.checkbox("Hide Prompt Tuned Models", value=False, disabled=False):
#     df = df[df['type'] != 'Prompt Fine-tuned']
# if st.sidebar.checkbox("Hide Fine Tuned Models", value=False, disabled=False):
#     df = df[df['type'] != 'Fine-tuned (FT)']
# if st.sidebar.checkbox("Hide Few-shot (FT)", value=False, disabled=False):
#     df = df[df['type'] != 'Few-shot FT']
# if st.sidebar.checkbox("Hide Few-shot (PT) (Aligned)", value=False, disabled=False):
#     df = df[df['type'] != 'Few-shot PT (Aligned)']
# if st.sidebar.checkbox("Hide Few-shot (PT) (Unaligned)", value=False, disabled=False):
#     df = df[df['type'] != 'Few-shot PT (Unaligned)']

# if show_only_baseline:
#     df = df[df['type'] != 'Robustness Enhanced']
#     df = df[df['type'] != 'Parameter Efficient']

df['f1'] = pd.to_numeric(df["f1"])
df['f1_lower'] = pd.to_numeric(df["f1_lower"])
df['f1_upper'] = pd.to_numeric(df["f1_upper"])

df.drop_duplicates(inplace=True)

df['dataset'] = df['dataset'].str.lower()

# visible_models = st.sidebar.multiselect(label="Visible model families", options=list(
#     df["model_family"].unique()), default=list(df["model_family"].unique()))
# hidden_models = set(df["model_family"].unique()).difference(visible_models)

# for hidden in hidden_models:
#     df = df[df['model_family'] != hidden]

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

color_map = {'Fine-tuned' : 'rgba(230, 97, 0, 0.4)',
             "Prompt Fine-tuned": 'rgba(246,190,0, 0.5)',
             'Few-shot Fine-tuned': 'rgba(26, 133, 255, 0.4)',
             'Few-shot Prompt Fine-tuned': 'rgba(64, 176, 166, 0.4)',
             'Zero-shot': 'rgba(75, 0, 146, 0.4)',
             'In-Context Learning': 'rgba(212, 17, 89, 0.4)',}
            #  'Robustness Enhanced FT': 'rgba(0, 0, 0, 0.4)',
            #  'Parameter Efficient FT': 'rgba(128,128,0, 0.4)',
            #  'Parameter Efficient PT': 'rgba(128,128,0, 0.4)'}

color_map_mpl = {'Fine-tuned' : 'darkorange',
                 'Prompt Fine-tuned': 'gold',
                 'Few-shot Fine-tuned': 'royalblue',
                 'Few-shot Prompt Fine-tuned': 'teal',
                 'Zero-shot': 'darkviolet',
                 'In-Context Learning': 'crimson',
                 'Robustness Enhanced Fine-tuned': 'black',
                 'Parameter Efficient Fine-tuned': 'darkgreen',
                 'Parameter Efficient Prompt Fine-tuned': 'brown'}


symbol_map = {'Fine-tuned' : "o",
              "Prompt Fine-tuned": "^",
              "Few-shot Fine-tuned": "D",
              "In-Context Learning": "P",
              "Few-shot Prompt Fine-tuned": "X",
              "Zero-shot": "s",
              "Robustness Enhanced Fine-tuned": "*",
              "Parameter Efficient Fine-tuned": "o",
              "Parameter Efficient Prompt Fine-tuned": "^",}

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
    
# seperate legend from plot and make sure it doesn't overlap with plot
# if not plot_config.plotly:
    # hide legend
            
    # handles, labels = fig.axes[0].get_legend_handles_labels()
    # # fig.axes[0].legend_.remove()
    # fig.legend(handles, labels, loc='lower center',
    #             ncol=2)
    # # hide plot
    # fig.axes[0].set_visible(False)


# make plot square
# if not plot_config.plotly:
#     fig.set_size_inches(8, 8)
    
# # increase x axis label font size
# fig.axes[0].xaxis.label.set_size(14)
# fig.axes[0].yaxis.label.set_size(14)

# # increase title font size
# fig.axes[0].title.set_size(18)

# increase overall legend size
# fig.axes[0].legend(prop={'size': 24}, loc='upper left')

# increase legend marker size
# fig.axes[0].legend(markerscale=1.25)

# increase tick label font size
# fig.axes[0].tick_params(axis='both', which='major', labelsize=16)

# file_name = st.text_input("Save figure as", f"{in_distribution_label}_to_{out_distribution_label}_mpl")
# if st.button("Save Plot") and not plot_config.plotly:
#     # save only plot and not legend
#     fig.savefig(f"images/{file_name}.pdf")
