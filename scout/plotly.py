import chart_studio.plotly as py
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scanpy as sc
from plotly.subplots import make_subplots

_layout = go.Layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
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


def seismic(zcenter):
    return [
        (0, "#00004C"),
        (zcenter * 0.5, "#0000E6"),
        (zcenter, "white"),
        ((1 - zcenter) * 0.5, "#FF0808"),
        (1, "#840000"),
    ]


def pval_histogram(df, x="pvals_adj", layout=_layout, nbins=20, fig_path=None):
    bins = np.linspace(0, 1, nbins + 1)
    counts, bins = np.histogram(df[x], bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    borders = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    _sum = counts.sum()
    proportions = [f"{counts[i]*100.0/_sum:.1f}%" for i in range(len(counts))]
    # fig = px.histogram(df, x=x, nbins=nbins)
    fig = px.bar(
        x=centers,
        y=counts,
        hover_data={
            "Bin": borders,
            "Proportion": proportions
        },
    )
    fig.update_layout(layout)
    fig.update_layout(xaxis_title=x.replace("_", " ").title(),
                      yaxis_title="Count",
                      bargap=0)
    fig.update_traces(marker=dict(line=dict(color="black", width=1)))

    if fig_path is not None:
        fig.write_image(fig_path)

    return fig


# Plotly
def violin(
    data,
    y,
    groupby=None,
    layer=None,
    scatter=True,
    box=True,
    mean_line=False,
    scatter_size=1.0,
    jitter=0.6,
    violin_colors=SC_DEFAULT_COLORS,
    layout=_layout,
    fig_path=None,
):
    if type(data) == sc.AnnData:
        if y in data.obs_keys():
            df = pd.DataFrame({
                y: data.obs[y],
                groupby: data.obs[groupby]
            },
                              index=data.obs_names)
        elif y in data.var_names:
            _y = (data[:, y].X.toarray().flatten() if layer is None else
                  data[:, y].layers[layer].toarray().flatten())
            df = pd.DataFrame({
                y: _y,
                groupby: data.obs[groupby]
            },
                              index=data.obs_names)
        else:
            assert (
                False
            ), f"Feature {y} not found in adata.var_names or adata.obs_keys()"
    else:
        df = data

    fig = go.Figure()
    if groupby is not None:
        for i, group in enumerate(df[groupby].unique()):
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
                ))

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
                name=group,
                line_color=violin_colors[0],
            ))

    fig.update_layout(layout)
    fig.update_layout(
        xaxis_title=groupby.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title() if y in data.obs_keys() else y,
    )
    if fig_path is not None:
        fig.write_image(fig_path)

    return fig


def marker_volcano(
    df,
    x="logFC",
    y="-log_pvals_adj",
    hue="log_mu_expression",
    significance_threshold=0.05,
    cmap="plasma",
    layout=_layout,
    fig_path=None,
):
    df["significant"] = df["pvals_adj"] <= significance_threshold

    fig = px.scatter(
        df.reset_index(),
        x=x,
        y=y,
        color=hue,
        symbol="significant",
        symbol_map={
            True: "circle",
            False: "x"
        },
        hover_name=df.index.name,
        color_continuous_scale=cmap,
        hover_data={
            x: ":.2f",
            y: ":.2f",
            hue: ":.2f",
            "significant": False
        },
        # labels={x: "Log2 FC", y: "-Log10 p-value ", hue: "log2 Mean Expression"},
    )
    fig.update_traces(
        marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")))
    fig.add_hline(
        y=-np.log10(significance_threshold),
        line_width=1,
        line_dash="dash",
        line_color=sc.pl.palettes.default_20[3],
    )

    fig.update_layout(layout)
    fig.update_layout(
        xaxis_title="log2FC" if x == "logFC" else x.replace("_", " ").title(),
        yaxis_title="- Log10 Adj. P-value"
        if y == "-log_pvals_adj" else x.replace("_", " ").title(),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        coloraxis_colorbar=dict(title="", ),
        annotations=[
            dict(
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
                textangle=-90,
            )
        ],
    )

    if fig_path is not None:
        fig.write_image(fig_path)

    return fig


def _create_categorical_row(adata, category, cmap=SC_DEFAULT_COLORS):
    c = adata.obs[category].cat.codes.values
    cats = adata.obs[category].cat.categories.values
    n_cats = len(cats)

    color_scale = [cmap[i] for i in range(n_cats)]

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Heatmap(
            z=[c.T],
            y=[category.title()],
            colorscale=color_scale,
            # colorbar = dict(title="Sample", tickvals=color_vals, ticktext = cats)
        ),
        row=1,
        col=1,
    )
    fig.update_traces(showscale=False)

    # Dummy to show the colors in the legend
    for i, cat in enumerate(cats):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[category.title()],
                showlegend=True,
                marker=dict(color=cmap[i], size=10),
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
        yaxis=dict(showgrid=False,
                   zeroline=False,
                   visible=True,
                   showticklabels=True),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            title_text="Categorical",
            x=1,
        ),
    )
    return fig


def heatmap(
    adata,
    var_names,
    categoricals=None,
    layer="logcentered",
    fig_path=None,
    layout=_layout,
):
    n_cats = len(categoricals)
    h_cat = 0.5
    n_vars = len(var_names)
    h_vars = 0.3
    r_cat = int(h_cat * 100.0 / (n_cats * h_cat + n_vars * h_vars))
    r_vars = 100 - r_cat
    height_ratios = [r_cat] * n_cats + [r_vars]
    cmaps = [SC_DEFAULT_COLORS, PLOTLY_DISCRETE_COLORS]

    fig = make_subplots(
        rows=len(categoricals) + 1,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=height_ratios,
        column_widths=[0.05, 0.95],
        horizontal_spacing=0.005,
    )

    if not f"dendrogram_barcode" in adata.uns.keys():
        adata.obs["barcode"] = pd.Categorical(adata.obs_names)
        sc.tl.dendrogram(adata, groupby="barcode", var_names=var_names)

    cell_order = adata.uns["dendrogram_barcode"]["categories_ordered"]

    gene_dendro = ff.create_dendrogram(adata[:, var_names].X.toarray().T,
                                       orientation="left")
    # Get min for the range
    x_max = max([max(trace_data.x) for trace_data in gene_dendro["data"]])

    for trace in gene_dendro["data"]:
        fig.add_trace(trace, row=len(categoricals) + 1, col=1)
        fig.update_traces(showlegend=False, line=dict(width=1.5))

    for i, categorical in enumerate(categoricals):
        subfig = _create_categorical_row(adata[cell_order, :],
                                         categorical,
                                         cmap=cmaps[i % len(cmaps)])["data"]
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

    fig.add_trace(
        go.Heatmap(z=z.T,
                   y=y_ticks,
                   colorscale=seismic(zcenter),
                   showlegend=False),
        row=len(categoricals) + 1,
        col=2,
    )

    if layout is None:
        layout = {}
    layout["height"] = (5 + len(var_names)) * 20 + 50
    fig.update_layout(layout)

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(
            orientation="h",
            entrywidth=70,
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
                ax:
                dict(
                    showgrid=False,
                    zeroline=False,
                    visible=False,
                    showticklabels=False,
                    range=[0, adata.n_obs],
                )
            })
        if ax[:5] == "yaxis":
            fig.update_layout({
                ax:
                dict(
                    showgrid=False,
                    zeroline=False,
                    visible=True,
                    showticklabels=True,
                )
            })

    fig.update_layout({
        # Dendro x-axis and y-axis -1 last axis i.e. axis7 if axis8 is last
        f"xaxis{last_axis-1}":
        dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            showticklabels=False,
            range=[10, x_max + 5],
            ticks="",
        ),
        f"yaxis{last_axis-1}":
        dict(
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
        f"yaxis{last_axis}":
        dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            showticklabels=False,
            ticks="",
        ),
    })

    if fig_path is not None:
        fig.write_image(fig_path)

    return fig
