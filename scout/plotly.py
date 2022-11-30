import scanpy as sc
import pandas as pd
import numpy as np

import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
import plotly

_layout = go.Layout(
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(t=10, b=10, l=10, r=10),
    width=500,
    height=500,
    autosize=False,
    # xaxis=dict(showgrid=False, zeroline=False, visible=True, showticklabels=True),
    # yaxis=dict(showgrid=False, zeroline=False, visible=True, showticklabels=True),
)

PLOTLY_DEFAULT_COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS
PLOTLY_DISCRETE_COLORS = px.colors.qualitative.Plotly
SC_DEFAULT_COLORS = sc.pl.palettes.default_20
NONE_COLOR = "#d3d3d3"

# Plotly
def violin(data, y, groupby=None, layer=None, scatter=True, box=True, mean_line=False,
    scatter_size=1.0, jitter=0.6, violin_colors=SC_DEFAULT_COLORS, layout=_layout, fig_path=None):
    if type(data) == sc.AnnData:
        if y in data.obs_keys():
            df = pd.DataFrame({y: data.obs[y], groupby: data.obs[groupby]}, index=data.obs_names)
        elif y in data.var_names:
            _y = data[:, y].X.toarray().flatten() if layer is None else data[:, y].layers[layer].toarray().flatten()
            df = pd.DataFrame({y: _y, groupby: data.obs[groupby]}, index=data.obs_names)
        else:
            assert False, f"Feature {y} not found in adata.var_names or adata.obs_keys()"
    else:
        df = data

    fig = go.Figure()
    if groupby is not None:
        for i, group in enumerate(df[groupby].unique()):
            fig.add_trace(go.Violin(
                y=df[df[groupby] == group][y], box_visible=box, meanline_visible=mean_line, points="all" if scatter else False, pointpos=0,
                marker=dict(size=scatter_size), jitter=jitter, name=group, line_color=violin_colors[i]
            ))

    else:
        fig.add_trace(go.Violin(
            y=df[y], box_visible=box, meanline_visible=mean_line, points="all" if scatter else False, pointpos=0,
            marker=dict(size=scatter_size), jitter=jitter, name=group, line_color=violin_colors[0]
        ))

    fig.update_layout(layout)
    fig.update_layout(
        xaxis_title=groupby.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title() if y in data.obs_keys() else y,
    )
    if fig_path is not None:
        fig.write_image(fig_path)

    return fig

def marker_volcano(df, x="logFC", y="-log_pvals_adj", hue="log_mu_expression", significance_threshold=0.05,
    cmap="plasma", layout=_layout, fig_path=None):
    df["significant"] = df["pvals_adj"] <= significance_threshold
    
    fig = px.scatter(
        df.reset_index(), x=x, y=y, color=hue, symbol="significant", symbol_map={True: "circle", False: "x"},
        hover_name=df.index.name, color_continuous_scale=cmap, 
        hover_data={x:":.2f", y:":.2f", hue:":.2f", "significant":False},
        # labels={x: "Log2 FC", y: "-Log10 p-value ", hue: "log2 Mean Expression"},
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")))
    fig.add_hline(
        y=-np.log10(significance_threshold), line_width=1, line_dash="dash",
        line_color=sc.pl.palettes.default_20[3]
    )

    fig.update_layout(layout)
    fig.update_layout(
        xaxis_title="log2FC" if x=="logFC" else x.replace("_", " ").title(),
        yaxis_title="- Log10 Adj. P-value" if y=="-log_pvals_adj" else x.replace("_", " ").title(),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        coloraxis_colorbar=dict(
            title="",
        ),
        annotations=[dict(
            x=0.99,
            align="right",
            valign="top",
            text=hue.replace("_", " ").title(),
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            # Parameter textangle allow you to rotate annotation how you want
            textangle=-90
        )]
    )


    if fig_path is not None:
        fig.write_image(fig_path)

    return fig