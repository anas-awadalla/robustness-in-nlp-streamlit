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
import matplotlib.pyplot as plt

"""
The following datasets are available along with thier sizes:
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
"""
dataset_to_size = {
    'squad': 10570,
    'squadshifts_nyt': 10065,
    'squadshifts_reddit': 9803,
    'squadshifts_new_wiki': 7938,
    'squadshifts_amazon': 9885,
    'RACE': 674, 
    'DROP': 1503, 
#     'TextbookQA': 1503,
    'BioASQ': 1504,
    'RelationExtraction': 2948,
    'NewsQA': 4212,
    'DuoRC.ParaphraseRC': 1501,
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

hide_zero_shot = st.sidebar.checkbox("Hide Zero Shot Models", value=False)
hide_few_shot = st.sidebar.checkbox("Hide Few Shot Models", value=False)
hide_icl = st.sidebar.checkbox("Hide In Context Learning Models", value=False)
hide_finetuned = st.sidebar.checkbox("Hide Finetuned Models", value=False)
logit_scaling = st.sidebar.checkbox("Apply Logit Scaling", value=False)

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
    
    try:
        with open(selected_file) as f:
            data = json.load(f)
    except:
        print(f"{model} prediction files not found")
        return

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
iid_bootstrap_f1 = defaultdict()
to_remove = []
for model in df['model_name'].unique():
    ood_output = bootstrap(
        model, dataset, n_samples_ood, num_iterations_ood)
    id_output = bootstrap(
        model, id_dataset, n_samples_iid, num_iterations_ood)
    if ood_output and id_output:
        ood_bootstrap_f1[model] = ood_output
        iid_bootstrap_f1[model] = id_output
    else:
        to_remove.append(model)

df = df[df['model_name'] not in to_remove]

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

ood_df = ood_df.drop(columns=['type', 'model_family', 'exact_match'])
iid_df = iid_df.rename(columns={"f1": "iid_f1"})
ood_df = ood_df.rename(columns={"f1": "ood_f1"})

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

if not logit_scaling:
    
    fig = px.scatter(dataset_df, x="iid_f1", y="ood_f1", color="model_family",
                     hover_data=["model_name", "type"], error_x="e_plus_iid", error_x_minus="e_minus_iid",
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
        z = np.polyfit(dataset_df[dataset_df['type'] == 'zeroshot']['iid_f1'],
                       dataset_df[dataset_df['type'] == 'zeroshot']['ood_f1'], 1)
        y_fit = np.poly1d(z)(dataset_df[dataset_df['type'] == 'zeroshot']['iid_f1'])

        line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(dataset_df[dataset_df['type'] == 'zeroshot']['ood_f1'] ,y_fit):0.3f}"
        fig.add_traces(go.Scatter(x=dataset_df[dataset_df['type'] == 'zeroshot']
                       ['iid_f1'], y=y_fit, name='Zero-Shot Fit:'+line_equation, mode='lines'))

    if not hide_few_shot:
        # Add trendline for few-shot models
        z = np.polyfit(dataset_df[dataset_df['type'] == 'fewshot']['iid_f1'],
                       dataset_df[dataset_df['type'] == 'fewshot']['ood_f1'], 1)
        y_fit = np.poly1d(z)(dataset_df[dataset_df['type'] == 'fewshot']['iid_f1'])

        line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(dataset_df[dataset_df['type'] == 'fewshot']['ood_f1'] ,y_fit):0.3f}"
        fig.add_traces(go.Scatter(x=dataset_df[dataset_df['type'] == 'fewshot']
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
    
    st.plotly_chart(fig, use_container_width=True)

else:
        fig, ax = plt.subplots()
        
        # From: https://github.com/mlfoundations/wise-ft/blob/master/src/scatter_plot.py
        def adjust_plot(ax, xrange, yrange, scaling, tick_freq=0.05):
            if type(tick_freq) is tuple:
                xtick_freq, ytick_freq = tick_freq[0], tick_freq[1]
            else:
                xtick_freq, ytick_freq = tick_freq, tick_freq

            if scaling == 'probit':
                h = scipy.stats.norm.ppf
            elif scaling == 'logit':
                h = lambda p: np.log(p / (1 - p))
            else:
                h = lambda p: p

            def transform(z):
                return [h(p) for p in z]

            tick_loc_x = [round(z, 2) for z in np.arange(xrange[0], xrange[1], xtick_freq)]
            ax.set_xticks(transform(tick_loc_x))
            ax.set_xticklabels([str(round(loc * 100)) for loc in tick_loc_x], fontsize=13)

            tick_loc_y = [round(z, 2) for z in np.arange(yrange[0], yrange[1], ytick_freq)]
            ax.set_yticks(transform(tick_loc_y))
            ax.set_yticklabels([str(round(loc * 100)) for loc in tick_loc_y], fontsize=13)

            ax.set_ylim(h(yrange[0]), h(yrange[1]))
            ax.set_xlim(h(xrange[0]), h(xrange[1]))

            ax.grid(linewidth=0.5)

            return transform
        
        dataset_df['ood_f1']/=100.0
        dataset_df['iid_f1']/=100.0
        
        y_range = (0.95 * min(dataset_df['ood_f1']), 1.05 * max(dataset_df['ood_f1']))
        x_range = (0.95 * min(dataset_df['iid_f1']), 1.05 * max(dataset_df['iid_f1']))
        
        try:
            transform = adjust_plot(ax, x_range, y_range, scaling='logit')
        except:
            raise Exception(f"{x_range[0]}{x_range[1]}{y_range[0]}{y_range[1]}")
        
        if not hide_zero_shot:
            zeroshot_xs = dataset_df[dataset_df["type"] == "zeroshot"]["iid_f1"]
            zeroshot_ys = dataset_df[dataset_df["type"] == "zeroshot"]["ood_f1"]

            # Add zero-shot
            ax.scatter(transform(zeroshot_xs), transform(zeroshot_ys),
                label=f"Zeroshot Models", marker='*', s=200, alpha=0.8, c='y',
            )

            ### Zero-shot linear fit
            sorted_pts = sorted(zip(zeroshot_xs, zeroshot_ys), key=lambda x: x[0])
            zeroshot_xs, zeroshot_ys = zip(*sorted_pts)
            z_zeroshot = np.polyfit(transform(zeroshot_xs), transform(zeroshot_ys), 1)
            y_linear_zeroshot = np.poly1d(z_zeroshot)(transform(zeroshot_xs))

            ax.plot(transform(zeroshot_xs), y_linear_zeroshot, "y-", label="Zeroshot Fit"+ f" $y={z_zeroshot[0]:0.3f}\;x{z_zeroshot[1]:+0.3f}$\n$R^2 = {r2_score(transform(zeroshot_ys),y_linear_zeroshot):0.3f}$")
           
        if not hide_finetuned:
            finetuned_xs = dataset_df[dataset_df["type"] == "finetuned"]["iid_f1"]
            finetuned_ys = dataset_df[dataset_df["type"] == "finetuned"]["ood_f1"]

            # Add fine-tuned
            ax.scatter(transform(finetuned_xs), transform(finetuned_ys),
                label=f"Finetuned Models", marker='D', s=100, alpha=0.8, c='g',
            )

            ### Fine-tuned linear fit
            sorted_pts = sorted(zip(finetuned_xs, finetuned_ys), key=lambda x: x[0])
            finetuned_xs, finetuned_ys = zip(*sorted_pts)
            z_finetuned = np.polyfit(transform(finetuned_xs), transform(finetuned_ys), 1)
            y_linear_finetuned = np.poly1d(z_finetuned)(transform(finetuned_xs))

            ax.plot(transform(finetuned_xs), y_linear_finetuned, "g-", label="Finetuned Fit"+ f" $y={z_finetuned[0]:0.3f}\;x{z_finetuned[1]:+0.3f}$\n$R^2 = {r2_score(transform(finetuned_ys),y_linear_finetuned):0.3f}$")
        
        if not hide_icl:
            finetuned_xs = dataset_df[dataset_df["type"] == "icl"]["iid_f1"]
            finetuned_ys = dataset_df[dataset_df["type"] == "icl"]["ood_f1"]

            # Add ICL
            ax.scatter(transform(finetuned_xs), transform(finetuned_ys),
                label=f"ICL Models", marker='D', s=100, alpha=0.8, c='r',
            )

            ### Fine-tuned linear fit
            sorted_pts = sorted(zip(finetuned_xs, finetuned_ys), key=lambda x: x[0])
            finetuned_xs, finetuned_ys = zip(*sorted_pts)
            z_finetuned = np.polyfit(transform(finetuned_xs), transform(finetuned_ys), 1)
            y_linear_finetuned = np.poly1d(z_finetuned)(transform(finetuned_xs))

            ax.plot(transform(finetuned_xs), y_linear_finetuned, "r-", label="ICL Fit"+ f" $y={z_finetuned[0]:0.3f}\;x{z_finetuned[1]:+0.3f}$\n$R^2 = {r2_score(transform(finetuned_ys),y_linear_finetuned):0.3f}$")
        
        if not hide_few_shot:
            finetuned_xs = dataset_df[dataset_df["type"] == "fewshot"]["iid_f1"]
            finetuned_ys = dataset_df[dataset_df["type"] == "fewshot"]["ood_f1"]

            # Add Fewshot
            ax.scatter(transform(finetuned_xs), transform(finetuned_ys),
                label=f"Fewshot Models", marker='*', s=100, alpha=0.8, c='b',
            )

            ### Fewshot linear fit
            sorted_pts = sorted(zip(finetuned_xs, finetuned_ys), key=lambda x: x[0])
            finetuned_xs, finetuned_ys = zip(*sorted_pts)
            z_finetuned = np.polyfit(transform(finetuned_xs), transform(finetuned_ys), 1)
            y_linear_finetuned = np.poly1d(z_finetuned)(transform(finetuned_xs))

            ax.plot(transform(finetuned_xs), y_linear_finetuned, "b-", label="Fewshot Fit"+ f" $y={z_finetuned[0]:0.3f}\;x{z_finetuned[1]:+0.3f}$\n$R^2 = {r2_score(transform(finetuned_ys),y_linear_finetuned):0.3f}$")
        
        ### y=x line
        ax.plot(min([ax.get_xlim(), ax.get_ylim()]), max([ax.get_xlim(), ax.get_ylim()]), '--', label="y=x")
        
        ax.legend(loc='best', prop={'size': 8})
        
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.title(f"Performance on {dataset} vs {id_dataset}")
        plt.xlabel(f"F1 Scores on {id_dataset}")
        plt.ylabel(f"F1 Scores on {dataset}")
        
        dataset_df['ood_f1']*=100.0
        dataset_df['iid_f1']*=100.0
        
        st.pyplot(fig)
          
        
dataset_df = dataset_df.rename(columns={"iid_f1": "id_f1"})
dataset_df = dataset_df.drop(columns=["iid_bootstrap_f1", "ood_bootstrap_f1"])
dataset_df = dataset_df.rename(columns={"e_minus_iid": "e_minus_id"})
dataset_df = dataset_df.rename(columns={"e_plus_iid": "e_plus_id"})


st.dataframe(dataset_df)
