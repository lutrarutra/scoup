from typing import Literal
import random

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
import scipy

from .defaults import layout as default_layout
from . import colors


def _legend(categories, colors, marker_size=10, marker_outline_width=None):
    fig = go.Figure()
    for i, cat in enumerate(categories):
        marker = dict(color=colors[i % len(colors)], size=marker_size)
        if marker_outline_width is not None:
            marker["line"] = dict(color="black", width=1)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                showlegend=True,
                marker=marker,
                mode="markers",
                name=f"{cat}",
            )
        )

    return fig


def _add_traces(to_figure, from_figure):
    for trace in from_figure.data:
        to_figure.add_trace(trace)

    return to_figure

def projection(
    adata, color=None, obsm_layer: str = "X_umap", color_layer="log1p",
    color_aggregate: Literal["abs", None] = "abs", fig_path=None,
    continuous_cmap="viridis", discrete_cmap="ScanPy Default",
    layout=default_layout, components=None, point_size=6.0, point_outline=1.0, point_opacity=1.0,
):
    fig = go.Figure()

    if type(discrete_cmap) == str:
        discrete_cmap = colors.get_discrete_colorscales()[discrete_cmap]

    if color is None:
        color_label = ""
        cmap = None
    else:
        if color in adata.obs_keys():
            color_label = color
            color = adata.obs[color]
            
            if not pd.api.types.is_numeric_dtype(color):
                cmap = discrete_cmap
                _order = color.unique().tolist()
                color = color.values
                cats = adata.obs[color_label].astype("category").cat.categories.tolist()
                cmap = [cmap[cats.index(cat) % len(cmap)] for cat in _order]
                fig = _add_traces(fig, _legend(
                    categories=adata.obs[color_label].astype("category").cat.categories.tolist(),
                    colors=discrete_cmap, marker_outline_width=1
                ))
            else:
                if continuous_cmap == "seismic":
                    zmin, zmax = np.quantile(color, [0.0, 1.0])
                    zcenter = abs(zmin) / (zmax - zmin)
                    cmap = colors.seismic(zcenter)
                else:
                    cmap = continuous_cmap

        else:
            if isinstance(color, str):
                color_label = color
                if color_layer == "log1p" or color_layer == "X" or color_layer == None:
                    color = adata.X[:, adata.var.index.get_loc(color)]
                elif color_layer in adata.layers.keys():
                    color = adata.layers[color_layer][:, adata.var.index.get_loc(color)]
                else:
                    assert False
                    
            elif isinstance(color, list):
                color_label = "Marker Score"
                if color_aggregate == "abs":
                    color = np.abs(adata[:, color].layers["logcentered"]).mean(1)
                elif color_aggregate == None:
                    color = adata[:, color].layers["logcentered"].mean(1)
            else:
                assert False

            if isinstance(color, scipy.sparse.csr_matrix):
                color = color.toarray()

            color = color.flatten()

            if continuous_cmap == "seismic":
                zmin, zmax = np.quantile(color, [0.0, 1.0])
                zcenter = abs(zmin) / (zmax - zmin)
                cmap = colors.seismic(zcenter)
            else:
                cmap = continuous_cmap

    axis_title = obsm_layer.replace("X_", "").replace("_", " ").upper()
    
    if (adata.obsm[obsm_layer].shape[1] == 2) or (components is not None and len(components) == 2):
        if components == None:
            components = (0, 1)
        
        df = pd.DataFrame(dict(
            x=adata.obsm[obsm_layer][:, components[0]],
            y=adata.obsm[obsm_layer][:, components[1]]
        ))

        if color is not None:
            df["color"] = color

        scatter = px.scatter(
            data_frame=df, x="x", y="y",
            color="color" if color is not None else None,
            color_discrete_sequence=cmap,
            color_continuous_scale=cmap,
            labels={
                "x": f"{axis_title} {components[0] + 1}",
                "y": f"{axis_title} {components[1] + 1}",
            },
        )
        scatter.update_traces(
            marker=dict(
                size=point_size, opacity=point_opacity, line=dict(color="black", width=point_outline)
            ),
            showlegend=False,
            hovertemplate=(
                "UMAP " + str(components[0] + 1) + ": %{x:.1f}<br>" + 
                "UMAP " + str(components[1] + 1) + ": %{y:.1f}"
            )
        )

        scatter.update_layout(showlegend=True)

        fig = _add_traces(scatter, fig)

    else:
        if components == None:
            components = (0, 1, 2)

        df = pd.DataFrame(dict(
            x=adata.obsm[obsm_layer][:, components[0]],
            y=adata.obsm[obsm_layer][:, components[1]],
            z=adata.obsm[obsm_layer][:, components[2]],
        ))

        if color is not None:
            df["color"] = color

        scatter = px.scatter_3d(
            data_frame=df, x="x", y="y", z="z",
            color="color" if color is not None else None,
            color_discrete_sequence=cmap,
            color_continuous_scale=cmap,
        )

        scatter.update_traces(
            marker=dict(
                size=point_size, opacity=point_opacity, line=dict(color="black", width=point_outline)
            ),
            showlegend=False,
            hovertemplate=(
                "UMAP " + str(components[0] + 1) + ": %{x:.1f}<br>" +
                "UMAP " + str(components[1] + 1) + ": %{y:.1f}<br>" +
                "UMAP " + str(components[2] + 1) + ": %{y:.1f}"
            )
        )

        scatter.update_layout(showlegend=True)

        # fig = _add_traces(fig, scatter)
        fig = _add_traces(scatter, fig)
        fig.update_layout(
            scene=go.layout.Scene(
                xaxis_title=f"{axis_title} {components[0] + 1}",
                yaxis_title=f"{axis_title} {components[1] + 1}",
                zaxis_title=f"{axis_title} {components[2] + 1}",
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    fig.update_layout(layout)

    fig.update_layout(
        legend=dict(
            title=color_label,
            y=0.5
        ),
        title=dict(
            text=color_label,
            xanchor="center",
            x=0.5,
            yanchor="top",
            y=0.99,
        )
    )

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
    )

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig