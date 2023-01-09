import plotly.express as px

import numpy as np
import scanpy as sc

from .defaults import layout as default_layout

def gsea_volcano(
    gsea_df, x="nes", y="-log10_fdr", hue="matched_fraction",
    cmap="viridis", significance_threshold=0.05, fig_path=None,
    layout=default_layout
):

    fig = px.scatter(
        gsea_df.reset_index(), x=x, y=y, color=hue, hover_name="Term",
        hover_data={x: ":.2f", y: ":.2f", hue: ":.2f"}, color_continuous_scale=cmap
    )

    fig.update_layout(layout)

    fig.update_traces(
        marker=dict(size=7, line=dict(width=1, color="Black"))
    )

    fig.add_hline(
        y=-np.log10(significance_threshold),
        line_width=1,
        line_dash="dash",
        line_color=sc.pl.palettes.default_20[3],
    )

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig
