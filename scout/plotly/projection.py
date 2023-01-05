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
    adata, obsm_layer: str = "X_umap", hue=None, hue_layer="log1p",
    hue_aggregate: Literal["abs", None] = "abs", fig_path=None,
    continuous_cmap="viridis", discrete_cmap="ScanPy Default",
    layout=default_layout, components=None,
):
    fig = go.Figure()

    if type(discrete_cmap) == str:
        discrete_cmap = colors.get_discrete_colorscales()[discrete_cmap]

    if hue is None:
        color = None
        hue_title = ""
        cmap = None
    else:
        if hue in adata.obs_keys():
            hue_title = hue
            color = adata.obs[hue].values
            if not pd.api.types.is_numeric_dtype(color):
                cmap = discrete_cmap
                fig = _add_traces(fig, _legend(
                    categories=adata.obs[hue].cat.categories.tolist(),
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
            if isinstance(hue, str):
                hue_title = hue
                if hue_layer == "log1p" or hue_layer == "X":
                    color = adata.X[:, adata.var.index.get_loc(hue)]
                elif hue_layer in adata.layers.keys():
                    color = adata.layers[hue_layer][:, adata.var.index.get_loc(hue)]
                else:
                    assert False
                    
            elif isinstance(hue, list):
                hue_title = "Marker Score"
                if hue_aggregate == "abs":
                    color = np.abs(adata[:, hue].layers["logcentered"]).mean(1)
                elif hue_aggregate == None:
                    color = adata[:, hue].layers["logcentered"].mean(1)
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
                size=6.0, opacity=1.0, line=dict(color="black", width=1.0)
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
                size=3, line=dict(color="black", width=1)
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
            title=hue_title,
            y=0.5
        ),
    )

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
    )

    if fig_path is not None:
        fig.write_image(fig_path)

    return fig