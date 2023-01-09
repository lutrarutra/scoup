import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import scanpy as sc

from .defaults import layout as default_layout

def dispersion_plot(
    adata, x="mu", y="cv2", log_x=True, log_y=True,
    layout=default_layout, fig_path=None
):
    add_hue = "highly_variable" in adata.var_keys()

    if add_hue:
        cmap = ["#636EFA", "#d3d3d3"] if adata.var["highly_variable"][0] else ["#d3d3d3", "#636EFA"]
    else:
        cmap = ["#636EFA"]

    fig = px.scatter(
        adata.var.reset_index(),
        x=x, y=y, log_x=log_x, log_y=log_y, color="highly_variable",
        color_continuous_scale=cmap, color_discrete_sequence=cmap,
        hover_name="index",
    )
    fig.update_traces(
        marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey"))
    )
    fig.update_layout(layout)
    fig.update_layout(xaxis_title="Log Mean Expression", yaxis_title="CV^2")
    if add_hue:
        fig.update_layout(legend_title="Highly Variable")

    if fig_path:
        fig.write_image(fig_path, scale=5)

    return fig


def qc_violin(adata, layout=default_layout, fig_path=None):
    violins = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    fig = make_subplots(rows=1, cols=3)
    x = adata.obs[violins]
    df = pd.DataFrame(x, columns=violins)

    for i, feature in enumerate(violins):
        fig.add_trace(
            go.Violin(
                name=feature.replace("_", " ").title(),
                y=df[feature],
                box_visible=True,
                points="all",
                pointpos=0,
                marker=dict(size=2),
                jitter=0.6,
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(layout)
    fig.update_layout(showlegend=False)

    if fig_path:
        fig.write_image(fig_path, scale=5)

    return fig


def mt_plot(adata, pct_counts_mt=5.0, layout=default_layout, fig_path=None):
    color = (adata.obs["pct_counts_mt"] > pct_counts_mt).values

    cmap = ["#d3d3d3", "#636EFA"] if color[0] else ["#636EFA", "#d3d3d3"]

    fig = px.scatter(
        adata.obs,
        x="total_counts",
        y="pct_counts_mt",
        color=color,
        color_discrete_sequence=cmap,
    )
    fig.update_traces(
        marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey"))
    )
    fig.update_layout(layout)
    fig.update_layout(
        xaxis_title="Total Counts",
        yaxis_title="MT%",
        legend_title_text=f"MT > {pct_counts_mt:.1f}%",
    )
    fig.add_hline(
        y=pct_counts_mt,
        line_width=1,
        line_dash="dash",
        line_color=sc.pl.palettes.default_20[3],
    )

    if fig_path:
        fig.write_image(fig_path, scale=5)

    return fig
