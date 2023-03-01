import numpy as np
import pandas as pd

import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .defaults import layout as default_layout
from . import colors
from .. import tools


def _create_categorical_row(adata, category, cmap=colors.SC_DEFAULT_COLORS):
    c = adata.obs[category].cat.codes.values
    cats = adata.obs[category].cat.categories.values
    n_cats = len(cats)
    color_scale = [cmap[i % len(cmap)] for i in range(n_cats)]

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Heatmap(
            z=[c.T], y=[category.title()],
            colorscale=color_scale,
        ),
        row=1,col=1,
    )
    fig.update_traces(showscale=False)

    # Dummy to show the colors in the legend
    for i, cat in enumerate(cats):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[category.title()],
                showlegend=True,
                marker=dict(color=cmap[i % len(cmap)], size=10),
                mode="markers",
                name=f"{category.title()}: {cat}",
                # legendgrouptitle_text=category, legendgroup = category,
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=True,
            showticklabels=False,
            range=[0, adata.n_obs],
        ),
        yaxis=dict(showgrid=False, zeroline=False, visible=True, showticklabels=True),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            # entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            title_text="Categorical",
            x=1,
        ),
    )
    return fig


def heatmap(
    adata, var_names, categoricals=None, layer="logcentered", fig_path=None,
    layout=default_layout, cluster_cells_by=None, cmap=None,
):
    if type(categoricals) is str:
        categoricals = [categoricals]

    n_cats = len(categoricals)
    h_cat = 0.5
    n_vars = len(var_names)
    h_vars = 0.3
    r_cat = int(h_cat * 100.0 / (n_cats * h_cat + n_vars * h_vars))
    r_vars = 100 - r_cat
    height_ratios = [r_cat] * n_cats + [r_vars]
    cmaps = [colors.SC_DEFAULT_COLORS, colors.PLOTLY_DISCRETE_COLORS]

    fig = make_subplots(
        rows=len(categoricals) + 1, cols=2, shared_xaxes=True, vertical_spacing=0.01,
        row_heights=height_ratios, column_widths=[0.05, 0.95], horizontal_spacing=0.005,
    )

    if cluster_cells_by is not None:
        if "barcode" not in adata.obs.columns:
            adata.obs["barcode"] = pd.Categorical(adata.obs_names)

        # Free Sort Cells, can take a while 5-10minutes depending on number of cells
        if cluster_cells_by == "barcode":
            if not cluster_cells_by in adata.uns.keys():
                tools.dendrogram(adata, groupby=cluster_cells_by, var_names=var_names)

            cell_order = adata.uns["dendrogram_barcode"]["categories_ordered"]
        else:
            cell_order = []
            for cell_type in adata.obs[cluster_cells_by].cat.categories.tolist():
                dendro = tools.dendrogram(
                    adata[adata.obs[cluster_cells_by] == cell_type, :].copy(), groupby="barcode",
                    var_names=var_names, inplace=False
                )
                _dendro = list(set(dendro["categories_ordered"]) & set(adata.obs_names))
                cell_order.extend(_dendro)

    else:
        cell_order = adata.obs.index

    gene_dendro = ff.create_dendrogram(
        adata[:, var_names].X.toarray().T, orientation="left"
    )

    # Get min for the range
    x_max = max([max(trace_data.x) for trace_data in gene_dendro["data"]])

    for trace in gene_dendro["data"]:
        fig.add_trace(trace, row=len(categoricals) + 1, col=1)
        fig.update_traces(showlegend=False, line=dict(width=1.5))

    for i, categorical in enumerate(categoricals):
        subfig = _create_categorical_row(adata[cell_order, :], categorical, cmap=cmaps[i % len(cmaps)])["data"]
        for trace in subfig:
            fig.add_trace(trace, row=i + 1, col=2)

    y_ticks = gene_dendro["layout"]["yaxis"]["tickvals"]
    dendro_order = gene_dendro["layout"]["yaxis"]["ticktext"]
    dendro_order = list(map(int, dendro_order))
    var_names_ordered = np.array(var_names)[dendro_order]

    if layer == "log1p" or layer == "X":
        z = adata[cell_order, var_names_ordered].X.toarray()
    else:
        z = adata[cell_order, var_names_ordered].layers[layer].toarray()

    zmin, zmax = np.quantile(z, [0.0, 1.0])
    zcenter = abs(zmin) / (zmax - zmin)
    if cmap is None:
        if "centered" in layer:
            colorscale = colors.seismic(zcenter)
        else:
            colorscale = "viridis"
    else:
        if cmap == "seismic":
            colorscale = colors.seismic(zcenter)
        else:
            colorscale = cmap

    fig.add_trace(
        go.Heatmap(z=z.T, y=y_ticks, showlegend=False, colorscale=colorscale),
        row=len(categoricals) + 1, col=2,
    )

    if layout is None:
        layout = {}

    layout["height"] = (5 + len(var_names)) * 20 + 20 * len(categoricals)
    fig.update_layout(layout)

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(
            orientation="h",
            # entrywidth=70,
            yanchor="bottom",
            x=0.02 + 0.005,
            y=1.02,
            xanchor="left",
            title_text="",
        ),
    )

    last_axis = 0
    for ax in fig["layout"]:
        if ax[:5] == "xaxis":
            last_axis += 1
            fig.update_layout({
                ax: dict(
                    showgrid=False,
                    zeroline=False,
                    visible=False,
                    showticklabels=False,
                    range=[0, adata.n_obs],
                )
            })
        if ax[:5] == "yaxis":
            fig.update_layout({
                ax: dict(
                    showgrid=False,
                    zeroline=False,
                    visible=True,
                    showticklabels=True,
                )
            })

    fig.update_layout(
        {
            # Dendro x-axis and y-axis -1 last axis i.e. axis7 if axis8 is last
            f"xaxis{last_axis-1}": dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                showticklabels=False,
                range=[10, x_max + 5],
                ticks="",
            ),
            f"yaxis{last_axis-1}": dict(
                showgrid=False,
                zeroline=False,
                visible=True,
                showticklabels=True,
                ticks="",
                tickmode="array",
                tickvals=y_ticks,
                ticktext=var_names_ordered,
                range=[-len(var_names) * 10, 0],
            ),
            # f"yaxis{last_axis-1}":dict(showgrid=False, zeroline=False, visible=True, showticklabels=True, range=[-len(var_names)*10, 0]),
            # Heatmap yaxis remove y ticks i.e. gene names as we have them in dendrogram
            f"yaxis{last_axis}": dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                showticklabels=False,
                ticks="",
            ),
        }
    )

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig