# a plotting class that can be used to plot data from a csv file and trendlines using both the matplotlib and plotly libraries
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import *


# make an enum for scaling options
class Scaling:
    LINEAR = "Linear"
    PROBIT = "Probit"
    LOGIT = "Logit"

@dataclass
class config:
    title: str
    x_label: str
    y_label: str
    color_map: dict # a dictionary of unique group_by column values to colors
    group_by: str = "type" # used for coloring the points, shapes, trendlines, and legend
    legend_title: str = "Adaptation Type" # the title of the legend
    plotly: bool = False
    symbol_map: dict = None # a dictionary of unique group_by column values to symbols
    plot_x_label: str = "In-Distribution F1 Score" # the x label of the plot
    plot_y_label: str = "Out-of-Distribution F1 Score" # the y label of the plot
    scaling_type: Scaling = Scaling.LINEAR
    auto_shrink_x: bool = True
    auto_shrink_y: bool = True
    show_legend: bool = True
    ticks_step: int = 10

class Plotter:
    @staticmethod
    def plot(df, config):
        def plot_mpl(df, config):
            plt.figure(dpi=1200)
            # plot using matplotlib
            fig, ax = plt.subplots()

            for group in sorted(df[config.group_by].unique()):
                # get the data for this group
                group_data = df[df[config.group_by] == group]
                # plot the data
                ax.scatter(group_data[f'{config.x_label} transformed'],
                           group_data[f'{config.y_label} transformed'],
                           label=group, alpha=0.3,
                           marker=config.symbol_map[group], s=80, color=config.color_map[group])

            ax.set_title(config.title)
            ax.set_xlabel(config.plot_x_label)
            ax.set_ylabel(config.plot_y_label)
            
            # set x and y axis ticks
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks_text)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks_text)
            
            # add y=x baseline based on the lowest ticks
            ax.plot(x_ticks if x_ticks[-1] < y_ticks[-1] else y_ticks, x_ticks if x_ticks[-1] < y_ticks[-1] else y_ticks, color='black', linestyle="dashed", linewidth=2, label="y=x")

            # add trendlines for each group in config.group_by
            for group in sorted(df[config.group_by].unique()):
                # get the data for this group
                group_data = df[df[config.group_by] == group]
                # get the trendline for this group
                y = Plotter.__get_trendline(group_data[f'{config.x_label} transformed'], group_data[f'{config.y_label} transformed'])
                # add the trendline to the plot
                ax.plot(group_data[f'{config.x_label} transformed'], y, linewidth=2, color=config.color_map[group])
                
            #### *** ASTETICS *** ####
            ax.grid(True)
            ax.set_axisbelow(True)
            if config.show_legend:
                ax.legend(loc='upper left', title=config.legend_title, fontsize="small")
        
            return fig

        def plot_plotly(df, config):
            # scatter plot using plotly
            fig = px.scatter(df, x=f'{config.x_label} transformed',
                             y=f'{config.y_label} transformed',
                             color=config.group_by,
                             color_discrete_map=config.color_map,
                             symbol=config.group_by,
                             hover_name="Model",
                             hover_data={"ID F1": True,
                                         "OOD F1": True,
                                         "ID F1 transformed": False,
                                         "OOD F1 transformed": False})
            fig.update_layout(title=config.title)
            fig.update_xaxes(title=config.plot_x_label)
            fig.update_yaxes(title=config.plot_y_label)

            # set x and y axis ticks            
            fig.update_xaxes(tickmode='array', tickvals=x_ticks, ticktext=x_ticks_text)
            fig.update_yaxes(tickmode='array', tickvals=y_ticks, ticktext=y_ticks_text)
            
            # add y=x baseline
            fig.add_trace(go.Scatter(x=x_ticks if x_ticks[-1] < y_ticks[-1] else y_ticks, y=x_ticks if x_ticks[-1] < y_ticks[-1] else y_ticks, mode='lines', line=dict(color='black', width=2, dash='dash'), name="y=x")) 

            # add trendlines for each group in config.group_by
            for group in sorted(df[config.group_by].unique()):
                # get the data for this group
                group_data = df[df[config.group_by] == group]
                # get the trendline for this group
                y = Plotter.__get_trendline(group_data[f'{config.x_label} transformed'], group_data[f'{config.y_label} transformed'])
                # add the trendline to the plot
                fig.add_trace(go.Scatter(x=group_data[f'{config.x_label} transformed'], y=y, mode='lines', line=dict(color=config.color_map[group], width=2), showlegend=False))
            
            ### **** ASTETICS **** ###
            fig.update_traces(marker=dict(size=12), line=dict(width=4))
            fig.update_xaxes(gridcolor='lightgrey', showline=True, linewidth=2, linecolor='black', mirror=True)
            fig.update_yaxes(gridcolor='lightgrey', showline=True, linewidth=2, linecolor='black', mirror=True)
            fig.update_layout(
                            showlegend=config.show_legend,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis_tickfont_size=14,
                            yaxis_tickfont_size=14,
                            title_x=0.5, 
                            legend=dict(yanchor="top", y=0.99,
                                        xanchor="left", x=0.01,
                                        traceorder="normal",
                                        bgcolor="White", 
                                        title=config.legend_title))

            return fig
        
        # generate ticks
        if config.auto_shrink_x:
            # get min and max of x axis
            x_min = df[config.x_label].min()
            x_max = df[config.x_label].max()
            x_ticks, x_ticks_text = Plotter.__get_ticks(x_min, x_max + config.ticks_step, config.scaling_type,config.ticks_step)
        else:
            x_ticks, x_ticks_text = Plotter.__get_ticks(0, 100 + config.ticks_step, config.scaling_type,config.ticks_step)
            
        if config.auto_shrink_y:
            # get min and max of y axis
            y_min = df[config.y_label].min()
            y_max = df[config.y_label].max()
            y_ticks, y_ticks_text = Plotter.__get_ticks(y_min, y_max + config.ticks_step, config.scaling_type, config.ticks_step)
        else:
            y_ticks, y_ticks_text = Plotter.__get_ticks(0, 100 + config.ticks_step, config.scaling_type, config.ticks_step)
        
        # scale x and y values
        df[f'{config.x_label} transformed'] = transform(df[config.x_label], config.scaling_type)
        df[f'{config.y_label} transformed'] = transform(df[config.y_label], config.scaling_type)
        
        return plot_plotly(df, config) if config.plotly else plot_mpl(df, config)
            
    @staticmethod   
    def __get_trendline(x, y):
        # mean, preds = bootstrap_ci(df, x, y)
        z = np.polyfit(x, y, 1)
        return np.poly1d(z)(x)
    
    @staticmethod
    def __get_ticks(min_val, max_val, scaling, step=10):
        tick_locs = [round(z) for z in np.arange(min_val, max_val, step)]
        tick_labels = [str(z) for z in tick_locs]
        return transform(tick_locs, scaling), tick_labels 
    
    