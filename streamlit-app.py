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

if scaling=="Linear":
    
    fig = px.scatter(dataset_df, x="iid_f1", y="ood_f1", color="model_family",
                     hover_data=["model_name", "type"], error_x="iid_f1_upper", error_x_minus="iid_f1_lower",
                     error_y="ood_f1_upper", error_y_minus="ood_f1_lower", title=f"Performance Comparison Between {pandas_id_dataset} and {pandas_dataset}",
                     labels=dict(iid_f1=f"F1 Score Performance on {pandas_id_dataset}", ood_f1=f"F1 Score Performance on {pandas_dataset}"))

    if not hide_finetuned:
        finetuned_df = dataset_df[dataset_df["type"] == "finetuned"]
        if len(finetuned_df) != 0:
            # Add trendline for finetuned models
            z = np.polyfit(finetuned_df['iid_f1'],
                           finetuned_df['ood_f1'], 1)
            y_fit = np.poly1d(z)(finetuned_df['iid_f1'])

            line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(finetuned_df['ood_f1'] ,y_fit):0.3f}"
            fig.add_traces(go.Scatter(x=finetuned_df
                           ['iid_f1'], y=y_fit, name='Fine-Tuned Fit:'+line_equation, mode='lines'))

    if not hide_zero_shot:
        zeroshot_df = dataset_df[dataset_df["type"] == "zeroshot"]
        if len(zeroshot_df) != 0:
            # Add trendline for zero-shot models
            z = np.polyfit(zeroshot_df['iid_f1'],
                           zeroshot_df['ood_f1'], 1)
            y_fit = np.poly1d(z)(zeroshot_df['iid_f1'])

            line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(zeroshot_df['ood_f1'] ,y_fit):0.3f}"
            fig.add_traces(go.Scatter(x=zeroshot_df
                           ['iid_f1'], y=y_fit, name='Zero-Shot Fit:'+line_equation, mode='lines'))

    if not hide_few_shot:
        fewshot_df = dataset_df[dataset_df["type"] == "fewshot"]
        if len(fewshot_df) != 0:
            # Add trendline for few-shot models
            z = np.polyfit(fewshot_df['iid_f1'],
                           fewshot_df['ood_f1'], 1)
            y_fit = np.poly1d(z)(fewshot_df['iid_f1'])

            line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(fewshot_df['ood_f1'] ,y_fit):0.3f}"
            fig.add_traces(go.Scatter(x=fewshot_df
                           ['iid_f1'], y=y_fit, name='Few-Shot Fit:'+line_equation, mode='lines'))

    if not hide_icl:
        icl_df = dataset_df[dataset_df["type"] == "icl"]
        if len(icl_df) != 0:
            # Add trendline for icl models
            z = np.polyfit(icl_df['iid_f1'],
                           icl_df['ood_f1'], 1)
            y_fit = np.poly1d(z)(icl_df['iid_f1'])

            line_equation = f" y={z[0]:0.3f}x{z[1]:+0.3f} -- R^2 = {r2_score(icl_df['ood_f1'] ,y_fit):0.3f}"
            fig.add_traces(go.Scatter(x=icl_df
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

            if scaling == 'Probit':
                h = scipy.stats.norm.ppf
            elif scaling == 'Logit':
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
            transform = adjust_plot(ax, x_range, y_range, scaling=scaling)
        except:
            raise Exception(f"{x_range[0]}{x_range[1]}{y_range[0]}{y_range[1]}")
        
        if not hide_zero_shot:
            zeroshot_df = dataset_df[dataset_df["type"] == "zeroshot"]
            if len(zeroshot_df) != 0:
                zeroshot_xs = zeroshot_df["iid_f1"]
                zeroshot_ys = zeroshot_df["ood_f1"]

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
            finetuned_df = dataset_df[dataset_df["type"] == "finetuned"]
            if len(finetuned_df) != 0:
                finetuned_xs = finetuned_df["iid_f1"]
                finetuned_ys = finetuned_df["ood_f1"]

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
            icl_df = dataset_df[dataset_df["type"] == "icl"]
            if len(icl_df) != 0:
                finetuned_xs = icl_df["iid_f1"]
                finetuned_ys = icl_df["ood_f1"]

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
            fewshot_df = dataset_df[dataset_df["type"] == "fewshot"]
            if len(fewshot_df) != 0:
                finetuned_xs = fewshot_df["iid_f1"]
                finetuned_ys = fewshot_df["ood_f1"]

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
        locs, labels = plt.xticks()
        ax.plot(locs, locs, '--', label="y=x")
        
        ax.legend(loc='best', prop={'size': 8})
        
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.title(f"Performance on {pandas_dataset} vs {pandas_id_dataset}")
        plt.xlabel(f"F1 Scores on {pandas_id_dataset}")
        plt.ylabel(f"F1 Scores on {pandas_dataset}")
        
        dataset_df['ood_f1']*=100.0
        dataset_df['iid_f1']*=100.0
        
        st.pyplot(fig)
          
        
dataset_df = dataset_df.rename(columns={"iid_f1": "id_f1"})
dataset_df = dataset_df.rename(columns={"iid_f1_upper": "id_f1_upper"})
dataset_df = dataset_df.rename(columns={"iid_f1_lower": "id_f1_lower"})


st.dataframe(dataset_df)
