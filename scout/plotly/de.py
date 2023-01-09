import plotly.express as px

import scanpy as sc
import numpy as np

from .defaults import layout as default_layout


def pval_histogram(df, x="pvals_adj", layout=default_layout, nbins=20, fig_path=None):
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
        hover_data={"Bin": borders, "Proportion": proportions},
    )
    fig.update_layout(layout)
    fig.update_layout(
        xaxis_title=x.replace("_", " ").title(), yaxis_title="Count", bargap=0
    )
    fig.update_traces(marker=dict(line=dict(color="black", width=1)))

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig

def marker_volcano(
    df, x="logFC", y="-log_pvals_adj", hue="log_mu_expression",
    significance_threshold=0.05, cmap="plasma", layout=default_layout, fig_path=None
):

    df["significant"] = df["pvals_adj"] <= significance_threshold

    fig = px.scatter(
        df.reset_index(), x=x, y=y, color=hue, symbol="significant",
        symbol_map={True: "circle", False: "x"}, hover_name=df.index.name,
        color_continuous_scale=cmap,
        hover_data={x: ":.2f", y: ":.2f", hue: ":.2f", "significant": False},
        labels={x: "Log2 FC", y: "-Log10 p-value ", hue: "log2 Mean Expression"},
    )

    fig.update_traces(marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")))

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
        if y == "-log_pvals_adj"
        else x.replace("_", " ").title(),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        coloraxis_colorbar=dict(
            title="",
        ),
        annotations=[
            dict(
                x=0.99, align="right", valign="top", text=hue.replace("_", " ").title(),
                showarrow=False, xref="paper", yref="paper", xanchor="left",
                # Parameter textangle allow you to rotate annotation how you want
                yanchor="middle", textangle=-90,
            )
        ],
    )

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig