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
    page_title="QA Robustness",
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

average_all_datasets = st.sidebar.checkbox("Average All Datasets", value=False)

dataset = st.sidebar.selectbox(
    'Dataset (Y-Axis)',
    list(dataset_to_size.keys()), disabled=average_all_datasets)

id_dataset = st.sidebar.selectbox(
    'Dataset (X-Axis)',
    list(dataset_to_size.keys()))


pandas_dataset = dataset
pandas_id_dataset = id_dataset

scaling = st.sidebar.selectbox("Graph Scaling", ['Linear', 'Logit', 'Probit'])
color_by_dataset = st.sidebar.checkbox("Color by Pretraining Dataset", value=False)

results_path = Path(".") / "results"

df = pd.read_csv(os.path.join(results_path.absolute(),
                 'roberta runs.csv'))
df_annotated = pd.read_csv(os.path.join(results_path.absolute(),"annotated_data.csv"))

# change type finetuned to Fine-tuned and zeroshot to Zero-shot and fewshot to Few-shot and icl to In-Context Learning
df['type'] = df['type'].apply(lambda x: x.replace('finetuned', 'Fine-tuned').replace('zeroshot', 'Zero-shot').replace('fewshot', 'Few-shot').replace('icl', 'In-Context Learning'))

# join df and df_annotated by model_name
df = df.merge(df_annotated, on='model_name')

original_df = df.copy()

filter_min_size=None
filter_max_size=None
filter_model_types=None
filter_finetune_types=None
filter_enhance_types=None

st.markdown("""Run a Query""")
filter_model_types = st.multiselect("Model Family", df["model_family"].unique().tolist())
filter_finetune_types = st.multiselect("Finetune Type", df["type"].unique().tolist())
filter_enhance_types = st.multiselect("Enhance Type", ["Adapters", "Robustness"])

is_filtered = filter_model_types or filter_finetune_types or filter_enhance_types

# Add a textbox for query name
query_name = st.text_input("Query Name (for labeling purposing)", "query")

hide_zero_shot = st.sidebar.checkbox("Hide Zero Shot Models", value=False)
hide_few_shot = st.sidebar.checkbox("Hide Few Shot Models", value=False)
hide_icl = st.sidebar.checkbox("Hide In Context Learning Models", value=False)
hide_finetuned = st.sidebar.checkbox("Hide Finetuned Models", value=False)

df['f1'] = pd.to_numeric(df["f1"])
df['f1_lower'] = pd.to_numeric(df["f1_lower"])
df['f1_upper'] = pd.to_numeric(df["f1_upper"])

df.drop_duplicates(inplace=True)

df['dataset'] = df['dataset'].str.lower()

# add a color column to the dataframe with the color based on type column in rgba format with colors aqua for finetuned and orange for few shot and purple for zero shot and green for in context learning and faint grey for other
color_map = {'RoBERTa Large (Fine-tuned)': 'rgba(230, 97, 0, 0.8)','Fine-tuned':'rgba(230, 97, 0, 0.8)', 'Few-shot': 'rgba(64, 176, 166, 0.8)', 'Zero-shot': 'rgba(75, 0, 146, 0.8)', 'In-Context Learning': 'rgba(212, 17, 89, 0.8)', 'Other Models': 'rgba(127, 127, 127, 0.5)', query_name: 'rgba(26, 133, 255, 0.8)'}
color_map_lines = {'RoBERTa Large (Fine-tuned)': 'rgba(93, 58, 155, 1)','Fine-tuned': 'rgba(93, 58, 155, 1)', 'Few-shot': 'rgba(255, 190, 106, 1)', 'Zero-shot': 'rgba(26, 255, 26, 1)', 'In-Context Learning': 'rgba(26, 133, 255, 1)', 'Other Models': 'rgba(0, 0, 0, 1)', query_name: 'rgba(212, 17, 89, 1)'}

visible_models = st.sidebar.multiselect(label="Visible model families", options=list(df["model_family"].unique()), default=list(df["model_family"].unique()))
hidden_models = set(df["model_family"].unique()).difference(visible_models)

for hidden in hidden_models:
    df = df[df['model_family'] != hidden]

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

if hide_finetuned:
    dataset_df = dataset_df[dataset_df['type'] != 'Fine-tuned']
    
if hide_zero_shot:
    dataset_df = dataset_df[dataset_df['type'] != 'Zero-shot']
    
if hide_few_shot:
    dataset_df = dataset_df[dataset_df['type'] != 'Few-shot']
    
if hide_icl:
    dataset_df = dataset_df[dataset_df['type'] != 'In-Context Learning']
    
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
yrange = [70, 75]
xrange = [94, 95]

if scaling == 'Linear':
    dataset_df['iid_f1_lower'] = transform(dataset_df['iid_f1_lower'], scaling=scaling)
    dataset_df['iid_f1_upper'] = transform(dataset_df['iid_f1_upper'], scaling=scaling)
    dataset_df['ood_f1_lower'] = transform(dataset_df['ood_f1_lower'], scaling=scaling)
    dataset_df['ood_f1_upper'] = transform(dataset_df['ood_f1_upper'], scaling=scaling)
else:
    # remove bounds
    dataset_df.drop(columns=['iid_f1_lower', 'iid_f1_upper', 'ood_f1_lower', 'ood_f1_upper'], inplace=True)

# Create a dictionary of model_family: pretraining dataset
dataset_map = {"bert": "bookcorpus+wikipedia", "bart": "bookcorpus+wikipedia", "minilm": "bookcorpus+wikipedia",  "albert": "bookcorpus+wikipedia", "roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "gpt": "webtext", "bidaf":"none (Word2Vec or ELMO)", "spanbert": "bookcorpus+wikipedia", "t5": "C4", "adapter-roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "adapter-bert": "bookcorpus+wikipedia", "xlm-roberta": "bookcorpus+wikipedia+cc-news+openwebtext+stories", "gpt-neo": "pile", "splinter": "wikipedia+bookcorpus", "opt":"pile", "gpt-j": "pile"}

# Add a pretrain_dataset column and iterate over the rows to add the pretrain_dataset
dataset_df['pretrain_dataset'] = dataset_df['model_family'].apply(lambda x: dataset_map[x])

# rename type Fine-tuned to RoBERTa Large (Fine-tuned)
dataset_df.loc[dataset_df['type'] == 'Fine-tuned', 'type'] = 'RoBERTa Large (Fine-tuned)'

hover_data = {"iid_f1": True, "ood_f1": True, "type": True, "model_family": True, "pretrain_dataset": True, "iid_f1_transformed": False, "ood_f1_transformed": False}

if not is_filtered:
    if scaling == 'Linear':
        # plot scatter by model family
        fig = px.scatter(dataset_df, x="iid_f1_transformed", y="ood_f1_transformed", color="color" if color_by_dataset else "type", hover_name="model_name", hover_data=hover_data, title=f"Performance comparison between {'SQuAD' if pandas_id_dataset=='squad' else pandas_id_dataset} and {'all datasets' if average_all_datasets else pandas_dataset}", labels=dict(iid_f1_transformed=f"F1 score performance on {'SQuAD' if pandas_id_dataset=='squad' else pandas_id_dataset}", ood_f1_transformed=f"F1 score performance on {'all datasets' if average_all_datasets else pandas_dataset}"), opacity=0.8, symbol="type", color_discrete_map=None if color_by_dataset else color_map_lines)

    else:
        fig = px.scatter(dataset_df, x="iid_f1_transformed", y="ood_f1_transformed", color="pretrain_dataset" if color_by_dataset else "type", hover_name="model_name", hover_data=hover_data, title=f"Performance comparison between {'SQuAD' if pandas_id_dataset=='squad' else pandas_id_dataset} and {'all datasets' if average_all_datasets else pandas_dataset}", labels=dict(iid_f1_transformed=f"F1 score performance on {'SQuAD' if pandas_id_dataset=='squad' else pandas_id_dataset}", ood_f1_transformed=f"F1 score performance on {'all datasets' if average_all_datasets else pandas_dataset}"), opacity=0.8, symbol="type", color_discrete_map=None if color_by_dataset else color_map_lines)


    # tick_loc_x = [round(z) for z in np.arange(xrange[0], xrange[1], 1)]

    # set x axis ticks
    # fig.update_xaxes(tickmode='array', tickvals=transform(tick_loc_x, scaling=scaling), ticktext=[str(z) for z in tick_loc_x])

    # tick_loc_y = [round(z) for z in np.arange(yrange[0], yrange[1], 1)]

    # set y axis ticks
    # fig.update_yaxes(tickmode='array', tickvals=transform(tick_loc_y, scaling=scaling), ticktext=[str(z) for z in tick_loc_y])

    # turn off autoscale
    fig.update_layout(autosize=False)

@st.cache()
def bootstrap_ci(df, n=1000):
    # A function to calculate the confidence interval for a linear regression model using bootstrap
    # x: independent variable
    # y: dependent variable
    # n: number of bootstrap samples
    # returns: mean of bootstrap
    preds = []
    for _ in range(n):
        bootstrap_df = df.sample(frac=1, replace=True)
        # Add a trendline with confidence intervals using bootstrap
        z = np.polyfit(bootstrap_df['iid_f1_transformed'],
                        bootstrap_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(df['iid_f1_transformed'])
        preds.append(y_fit)

    return np.array(preds).mean(axis=0), preds
    
    
if not hide_finetuned and not is_filtered:
    finetuned_df = dataset_df[dataset_df["type"] == "RoBERTa Large (Fine-tuned)"]
    # sort by iid_f1_transformed
    finetuned_df.sort_values(by=['iid_f1_transformed'], inplace=True)

    if len(finetuned_df) != 0:
        mean, preds = bootstrap_ci(finetuned_df)

        z = np.polyfit(finetuned_df["iid_f1_transformed"], finetuned_df["ood_f1_transformed"], 1)
        y_fit = np.poly1d(z)(finetuned_df["iid_f1_transformed"])
        # line_equation = f" y={z[0]:0.2f}x{z[1]:+0.2f} -- R^2 = {r2_score(finetuned_df['ood_f1_transformed'] ,y_fit):0.2f}"
        fig.add_traces(go.Scatter(x=finetuned_df['iid_f1_transformed'], y=y_fit, name='RoBERTa Large Fine-Tuned Fit', mode='lines', line_color=color_map_lines['Fine-tuned'], showlegend=False))

        # lower_bound = mean-(2*np.std(np.array(preds),axis=0))
        # upper_bound = mean+(2*np.std(np.array(preds),axis=0))

        # # Add a shaded region for the confidence interval the same fill color as the line
        # fig.add_trace(go.Scatter(x=finetuned_df['iid_f1_transformed'], y=lower_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', line_color='rgba(0,0,0,0)', showlegend=False))
        # fig.add_trace(go.Scatter(x=finetuned_df['iid_f1_transformed'], y=upper_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', fill='tonexty', line_color='rgba(0,0,0,0)', showlegend=False))


if not hide_zero_shot and not is_filtered:
    zeroshot_df = dataset_df[dataset_df["type"] == "Zero-shot"]
    # sort by iid_f1_transformed
    zeroshot_df.sort_values(by=['iid_f1_transformed'], inplace=True)
    if len(zeroshot_df) != 0:
        mean, preds = bootstrap_ci(zeroshot_df)
        # Add trendline for zero-shot models
        z = np.polyfit(zeroshot_df['iid_f1_transformed'],
                        zeroshot_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(zeroshot_df['iid_f1_transformed'])

        # line_equation = f" y={z[0]:0.2f}x{z[1]:+0.2f} -- R^2 = {r2_score(zeroshot_df['ood_f1_transformed'] ,y_fit):0.2f}"
        # set line color to color_map['zeroshot']
        fig.add_traces(go.Scatter(x=zeroshot_df
                        ['iid_f1_transformed'], y=y_fit, name='Zero-Shot Fit', mode='lines', line_color=color_map_lines['Zero-shot']))

        lower_bound = mean-(2*np.std(np.array(preds),axis=0))
        upper_bound = mean+(2*np.std(np.array(preds),axis=0))

        fig.add_trace(go.Scatter(x=zeroshot_df['iid_f1_transformed'], y=lower_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=zeroshot_df['iid_f1_transformed'], y=upper_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', fill='tonexty', line_color='rgba(0,0,0,0)', showlegend=False))

if not hide_few_shot and not is_filtered:
    fewshot_df = dataset_df[dataset_df["type"] == "Few-shot"]
    # sort by iid_f1_transformed
    fewshot_df.sort_values(by=['iid_f1_transformed'], inplace=True)
    if len(fewshot_df) != 0:
        mean, preds = bootstrap_ci(fewshot_df)
        # Add trendline for few-shot models
        z = np.polyfit(fewshot_df['iid_f1_transformed'],
                        fewshot_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(fewshot_df['iid_f1_transformed'])

        # line_equation = f" y={z[0]:0.2f}x{z[1]:+0.2f} -- R^2 = {r2_score(fewshot_df['ood_f1_transformed'] ,y_fit):0.2f}"
        fig.add_traces(go.Scatter(x=fewshot_df
                        ['iid_f1_transformed'], y=y_fit, name='Few-Shot Fit', mode='lines', line_color=color_map_lines['Few-shot']))

        lower_bound = mean-(2*np.std(np.array(preds),axis=0))
        upper_bound = mean+(2*np.std(np.array(preds),axis=0))

        fig.add_trace(go.Scatter(x=fewshot_df['iid_f1_transformed'], y=lower_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=fewshot_df['iid_f1_transformed'], y=upper_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', fill='tonexty', line_color='rgba(0,0,0,0)', showlegend=False))

if not hide_icl and not is_filtered:
    icl_df = dataset_df[dataset_df["type"] == "In-Context Learning"]
    # sort by iid_f1_transformed
    icl_df.sort_values(by=['iid_f1_transformed'], inplace=True)
    if len(icl_df) != 0:
        mean, preds = bootstrap_ci(icl_df)
        # Add trendline for icl models
        z = np.polyfit(icl_df['iid_f1_transformed'],
                        icl_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(icl_df['iid_f1_transformed'])

        # line_equation = f" y={z[0]:0.2f}x{z[1]:+0.2f} -- R^2 = {r2_score(icl_df['ood_f1_transformed'] ,y_fit):0.2f}"
        fig.add_traces(go.Scatter(x=icl_df
                        ['iid_f1_transformed'], y=y_fit, name='In-Context Learning Fit', mode='lines', line_color=color_map_lines['In-Context Learning']))

        # lower_bound = mean-(2*np.std(np.array(preds),axis=0))
        # upper_bound = mean+(2*np.std(np.array(preds),axis=0))

        # fig.add_trace(go.Scatter(x=icl_df['iid_f1_transformed'], y=lower_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', line_color='rgba(0,0,0,0)', showlegend=False))
        # fig.add_trace(go.Scatter(x=icl_df['iid_f1_transformed'], y=upper_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', fill='tonexty', line_color='rgba(0,0,0,0)', showlegend=False))


if is_filtered:
    query_df = dataset_df.copy()
    if filter_model_types:
        query_df = dataset_df[dataset_df["model_family"].isin(filter_model_types)]

    if filter_finetune_types:
        query_df = query_df[query_df["type"].isin(filter_finetune_types)]

    if "Adapters" in filter_enhance_types:
        query_df = query_df.loc[query_df['is_adapter'] == True]

    # same for is_robust
    if "Robustness" in filter_enhance_types:
        query_df = query_df.loc[query_df['is_robust'] == True]

    # sort by iid_f1_transformed
    query_df.sort_values(by=['iid_f1_transformed'], inplace=True)

    filtered_df = dataset_df.loc[~dataset_df['model_name'].isin(query_df['model_name'])]

    filtered_df.sort_values(by=['iid_f1_transformed'], inplace=True)


    dataset_df.sort_values(by=['model_name'], inplace=True)

    # alter type column for the dataset_df based on the model_name to none
    dataset_df['type'].loc[~dataset_df['model_name'].isin(query_df['model_name'])] = 'Other Models'
    dataset_df['type'].loc[dataset_df['model_name'].isin(query_df['model_name'])] = query_name

    # plot scatter by model family
    fig = px.scatter(dataset_df, x="iid_f1_transformed", y="ood_f1_transformed", color="pretrain_dataset" if color_by_dataset else "type", hover_name="model_name", hover_data=hover_data, title=f"Performance Comparison Between {pandas_id_dataset} and {'All Datasets' if average_all_datasets else pandas_dataset}", labels=dict(iid_f1_transformed=f"F1 Score Performance on {pandas_id_dataset}", ood_f1_transformed=f"F1 Score Performance on {'All Datasets' if average_all_datasets else pandas_dataset}"), symbol="type", color_discrete_map=None if color_by_dataset else color_map)
    if len(query_df) > 1:
        mean, preds = bootstrap_ci(query_df)
        # Add trendline for icl models
        z = np.polyfit(query_df['iid_f1_transformed'],
                        query_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(query_df['iid_f1_transformed'])

        # line_equation = f" y={z[0]:0.2f}x{z[1]:+0.2f} -- R^2 = {r2_score(query_df['ood_f1_transformed'] ,y_fit):0.2f}"
        # set line color to black
        fig.add_traces(go.Scatter(x=query_df['iid_f1_transformed'], y=y_fit, name=f"{query_name or 'Query'} Fit", mode='lines', line_color=color_map_lines[query_name]))

        # lower_bound = mean-(2*np.std(np.array(preds),axis=0))
        # upper_bound = mean+(2*np.std(np.array(preds),axis=0))

        # fig.add_trace(go.Scatter(x=query_df['iid_f1_transformed'], y=lower_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', line_color='rgba(0,0,0,0)', showlegend=False))
        # fig.add_trace(go.Scatter(x=query_df['iid_f1_transformed'], y=upper_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', fill='tonexty', line_color='rgba(0,0,0,0)', showlegend=False))


    # Plot all other models
    if len(dataset_df) != 0:
        # add the scatter plot for the dataset_df with out the query_df models
        mean, preds = bootstrap_ci(filtered_df)
        # Add trendline for icl models
        z = np.polyfit(filtered_df['iid_f1_transformed'],
                        filtered_df['ood_f1_transformed'], 1)
        y_fit = np.poly1d(z)(filtered_df['iid_f1_transformed'])

        # line_equation = f" y={z[0]:0.2f}x{z[1]:+0.2f} -- R^2 = {r2_score(dataset_df['ood_f1_transformed'] ,y_fit):0.2f}"
        fig.add_traces(go.Scatter(x=filtered_df
                        ['iid_f1_transformed'], y=y_fit, name='All Models Fit', mode='lines', line_color=color_map_lines['Other Models']))
        # lower_bound = mean-(2*np.std(np.array(preds),axis=0))
        # upper_bound = mean+(2*np.std(np.array(preds),axis=0))
        # fig.add_trace(go.Scatter(x=filtered_df['iid_f1_transformed'], y=lower_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', line_color='rgba(0,0,0,0)', showlegend=False))
        # fig.add_trace(go.Scatter(x=filtered_df['iid_f1_transformed'], y=upper_bound, mode='lines', fillcolor='rgba(0,0,0,0.2)', fill='tonexty', line_color='rgba(0,0,0,0)', showlegend=False))


    # Rename legend title
    fig.update_layout(legend_title_text="Pretraining Dataset" if color_by_dataset else "Adaptation Type")

    

    # turn off autoscale
    fig.update_layout(autosize=True)

fig.update_traces(line=dict(width=4))


# tick_loc_x = [round(z) for z in np.arange(xrange[0], xrange[1], 0.1)]

# # set x axis ticks
# fig.update_xaxes(tickmode='array', tickvals=transform(tick_loc_x, scaling=scaling), ticktext=[str(z) for z in tick_loc_x])

# tick_loc_y = [round(z) for z in np.arange(yrange[0], yrange[1], 0.1)]

# # set y axis ticks
# fig.update_yaxes(tickmode='array', tickvals=transform(tick_loc_y, scaling=scaling), ticktext=[str(z) for z in tick_loc_y])

# Plot y=x line using tick values
# fig.add_trace(go.Line(x=transform(tick_loc_y), y=transform(tick_loc_y), mode='lines', name='y = x', line_dash="dash", line_color="black ",hoverinfo='none', line=dict(width=4)))
# Set plotly background color to transparent
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
# Set plotly grid lines to light grey
fig.update_xaxes(gridcolor='lightgrey', zerolinecolor='lightgrey', zerolinewidth=1)
fig.update_yaxes(gridcolor='lightgrey', zerolinecolor='lightgrey', zerolinewidth=1)

# set title only font size
fig.update_layout(font=dict(size=12))
fig.update_layout(title_font=dict(size=20))
fig.update_traces(marker=dict(size=12))
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
fig.update_layout(legend_title_text="")


# attached = st.checkbox("Attach Legend")
fig.update_layout(legend=dict( yanchor="top", y=0.99, xanchor="left",
                                x=0.01, traceorder="normal", bgcolor="White"))

if st.checkbox("Horizontal Legend"):
    fig.update_layout(legend=dict(orientation="h"))
    # detach legend from plot
    fig.update_layout(legend=dict(x=0, y=-0.8, yanchor="top"))

if st.checkbox("Hide Legend"):
    # hide legend from plot
    fig.update_layout(showlegend=False)

# Set all text to black
fig.update_layout(font=dict(color="black"))

st.plotly_chart(fig, use_container_width=True)

fig.update_layout(font=dict(family="sans-serif"))
fig.update_layout(font=dict(size=72))
fig.update_layout(title_font=dict(size=72))
fig.update_traces(marker=dict(size=64))
fig.update_layout(xaxis_tickfont_size=54, yaxis_tickfont_size=54)
fig.update_traces(line=dict(width=10))

fig.update_layout(xaxis_tickfont_size=54, yaxis_tickfont_size=54)
# add padding between axes labels and tick labels
fig.update_xaxes(title_standoff = 50)
fig.update_yaxes(title_standoff = 50)

# set background color to transparent
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

if st.checkbox("Remove Legend on Save"):
    fig.update_layout(showlegend=False)

if st.checkbox("Save Plot as Square"):
    if pandas_dataset in ['NaturalQuestionsShort', 'DuoRC.ParaphraseRC']:
        # decrease title font size
        fig.update_layout(title_font=dict(size=58))
    fig.update_layout(height=1950, width=1950)
else:
    fig.update_layout(height=1080, width=1920)

fig.write_image("plot.pdf")


with open("plot.pdf", "rb") as pdf:
    download_btn = st.download_button("Download Plot",
             data=pdf,
             file_name="plot.pdf",
             mime="pdf")

# remove transformed f1 cols
dataset_df.drop(["iid_f1_transformed", "ood_f1_transformed"], axis=1, inplace=True)

dataset_df = dataset_df.rename(columns={"iid_f1": "id_f1"})
dataset_df = dataset_df.rename(columns={"iid_f1_upper": "id_f1_upper"})
dataset_df = dataset_df.rename(columns={"iid_f1_lower": "id_f1_lower"})


st.dataframe(dataset_df)
