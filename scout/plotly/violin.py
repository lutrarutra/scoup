import plotly.graph_objects as go

import pandas as pd
import scanpy as sc
import scipy

from .defaults import layout as default_layout
from . import colors

def violin(
    data, y, groupby=None, layer=None, scatter=True, box=True,
    mean_line=False, scatter_size=1.0, jitter=0.6,
    violin_colors=colors.SC_DEFAULT_COLORS, layout=default_layout, fig_path=None,
):
    if isinstance(data, sc.AnnData):
        if y in data.obs_keys():
            _y = data.obs[y]

        elif y in data.var_names:
            if layer is None:
                _y = data[:, y].X

            else:
                _y = data[:, y].layers[layer]

            if isinstance(_y, scipy.sparse.csr_matrix):
                _y = _y.toarray()

            _y = _y.flatten()
        else:
            assert (
                False
            ), f"Feature {y} not found in adata.var_names or adata.obs_keys()"

        _groupby = data.obs[groupby] if groupby is not None else None
        df = pd.DataFrame(
            data={y: _y, groupby: _groupby},
            index=data.obs_names
        )
    else:
        df = data

    fig = go.Figure()
    fig.update_layout(layout)

    if groupby is not None:
        for i, group in enumerate(df[groupby].cat.categories.tolist()):
            fig.add_trace(
                go.Violin(
                    y=df[df[groupby] == group][y],
                    box_visible=box,
                    meanline_visible=mean_line,
                    points="all" if scatter else False,
                    pointpos=0,
                    marker=dict(size=scatter_size),
                    jitter=jitter,
                    name=group,
                    line_color=violin_colors[i],
                )
            )

        fig.update_layout(
            legend=dict(title=groupby.replace("_", "").title(), y=0.5),
            xaxis_showticklabels=True,
            xaxis_title=groupby.replace("_", " ").title(),
        )

    else:
        fig.add_trace(
            go.Violin(
                y=df[y],
                box_visible=box,
                meanline_visible=mean_line,
                points="all" if scatter else False,
                pointpos=0,
                marker=dict(size=scatter_size),
                jitter=jitter,
                line_color=violin_colors[0],
            )
        )
        fig.update_layout(
            xaxis_showticklabels=False,
            xaxis_title=""
        )

    fig.update_layout(
        yaxis_title=y.replace("_", " ").title() if y in data.obs_keys() else y,
    )

    # if groupby is not None:
    #     fig.update_layout(
    #         # xaxis_title = groupby.replace("_", " ").title(),
    #         legend=dict(title=groupby.replace("_", "").title(), y=0.5),
    #     )

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig
