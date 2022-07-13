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
    page_title="Annotated Example Plot",
    page_icon="",
)

results_path = Path(".") / "pages"

df = pd.read_csv(os.path.join(results_path.absolute(), "examples.csv"))

# out of distribution df
ood_df = df[df["dataset"] == "out-of-distribution"]
id_df = df[df["dataset"] == "squad"]

# create a new df by adding f1_ood from ood_df to id_df based on model_name
id_df["f1_ood"] = id_df.apply(lambda row: ood_df[ood_df["model_name"] == row["model_name"]]["f1"].values[0], axis=1)
dataset_df = id_df
color_map = {'Robust Model': 'rgba(230, 97, 0, 0.8)', 'Baseline Model': 'lightsteelblue'}

fig = px.scatter(dataset_df, x="f1",
                 y="f1_ood",
                 color="type",
                 hover_name="model_name",
                 title="Hypothetical graph illustrating effective robustness",
                 labels=dict(f1="F1 score on SQuAD", 
                             f1_ood="F1 score on OOD test set"),
                 symbol="type",
                 color_discrete_map=color_map,
                )

yrange = [0, 110]
xrange = [0, 110]

finetuned_df = dataset_df[dataset_df["type"] == "Baseline Model"].reset_index(drop=True)
# sort by f1
finetuned_df.sort_values(by=['f1'], inplace=True)

z = np.polyfit(finetuned_df["f1"], finetuned_df["f1_ood"], 1)
y_fit = np.poly1d(z)(finetuned_df["f1"])
fig.add_traces(go.Scatter(x=finetuned_df['f1'], y=y_fit, name='Baseline Model Fit', mode='lines', line=dict(color=color_map['Baseline Model']), showlegend=False))

# Draw an annotation arrow to the Base Model Fit line
# fig.add_annotation(go.layout.Annotation(
#     x=finetuned_df['f1'][1],
#     y=y_fit[1],
#     text='Hypothetical Baseline Models',
#     xref='x',
#     yref='y',
#     showarrow=True,
#     arrowhead=4,
#     arrowsize=1,
#     arrowwidth=2,
#     ax=0,
#     ay=-50,
#     font=dict(
#         size=12,
#     )
# ))

zeroshot_df = dataset_df[dataset_df["type"] == "Robust Model"].reset_index(drop=True)
# sort by f1
zeroshot_df.sort_values(by=['f1'], inplace=True)

# Add trendline for zero-shot models
z = np.polyfit(zeroshot_df['f1'],
                zeroshot_df['f1_ood'], 1)
y_fit = np.poly1d(z)(zeroshot_df['f1'])

# set line color to color_map['zeroshot']
fig.add_traces(go.Scatter(x=zeroshot_df
                ['f1'], y=y_fit, name='Robust Model Fit', mode='lines', line=dict(color=color_map['Robust Model']), showlegend=False))

# Make annotation point downwards and add an arrow to annotation
# fig.add_annotation(go.layout.Annotation(
#     x=zeroshot_df['f1'][4],
#     y=y_fit[4]+3,
#     text='Hypothetical Robust Models',
#     xref='x',
#     yref='y',
#     showarrow=True,
#     arrowhead=4,
#     arrowsize=1,
#     arrowwidth=2,
#     ax=0,
#     ay=-50,
#     font=dict(
#         size=12,
#     )
# ))

# Draw arrow between two models and put text in the middle
fig.add_annotation(go.layout.Annotation(
    x=80,
    y=74.5,
    showarrow=True,
    arrowhead=4,
    arrowsize=1,
    arrowwidth=14,
    arrowcolor='black',
    ax=0,
    ay=588,
    font=dict(
        size=12,
    ),
))


# Add a text box to the plot
fig.add_annotation(go.layout.Annotation(
    x=86,
    y=67,
    text='Effective Robustness',
    showarrow=False,
    font=dict(
        size=72,
    ),
))

# remove legend title
fig.update_layout(legend_title_text="")

tick_loc_x = [round(z) for z in np.arange(65, 95, 5)]
tick_loc_y = [round(z) for z in np.arange(65, 95, 5)]

fig.update_traces(line=dict(width=10))

# Plot y=x line using tick values
fig.add_trace(go.Line(x=tick_loc_y, y=tick_loc_y, mode='lines', name='y = x', line_dash="dash", line_color="black",hoverinfo='none', line=dict(width=4)))

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

# Put legend in the top left corner of the plot as a two column table
fig.update_layout(legend=dict( yanchor="top", y=0.99, xanchor="left",
                              x=0.01, traceorder="normal", bgcolor="White"))

fig.update_layout(font=dict(color="black"))

st.plotly_chart(fig, use_container_width=True)

fig.update_layout(xaxis_tickfont_size=54, yaxis_tickfont_size=54)
# add padding between axes labels and tick labels
fig.update_xaxes(title_standoff = 50)
fig.update_yaxes(title_standoff = 50)

if st.checkbox("Save Plot as Square"):
    fig.update_layout(height=1920, width=1920)
else:
    fig.update_layout(height=3840, width=2160)

# fig set font size of legend text
fig.update_layout(legend_title_text="", legend=dict(title=dict(text="", font=dict(size=24))))

fig.update_layout(font=dict(family="sans-serif"))
fig.update_layout(font=dict(size=72))
fig.update_layout(title_font=dict(size=76))
fig.update_traces(marker=dict(size=60))
fig.update_layout(xaxis_tickfont_size=54, yaxis_tickfont_size=54)
fig.update_layout(legend= {'itemsizing': 'constant'})

# set background color to transparent
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.write_image("plot.pdf")

with open("plot.pdf", "rb") as pdf:
    download_btn = st.download_button("Download Plot",
             data=pdf,
             file_name="plot.pdf",
             mime="pdf")