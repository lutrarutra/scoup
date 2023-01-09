from typing import Literal

import plotly.express as px

import pandas as pd
import numpy as np

from .defaults import layout as default_layout
from .. import tools

def pca_explain_var(
    adata, plot_type: Literal["Cumulative", "Bar", "Line", "Area"] = "Cumulative",
    n_pcs=None, layout=default_layout, fig_path=None
):
    if n_pcs is None:
        n_pcs = adata.uns["pca"]["variance_ratio"].shape[0]

    if plot_type == "Bar":
        fig = px.bar(
            x=range(1, n_pcs + 1),
            y=adata.uns["pca"]["variance_ratio"][:n_pcs],
            labels={"x": f"PC", "y": f"Variance Ratio"},
        )
    else:
        y = adata.uns["pca"]["variance_ratio"][:n_pcs]

        _plot = None
        if plot_type == "Line":
            _plot = px.line
        elif plot_type == "Area":
            _plot = px.area
        elif plot_type == "Cumulative":
            _plot = px.area
            y = np.cumsum(y)

        df = pd.DataFrame({"PC": range(1, n_pcs + 1), "Variance Ratio": y})
        fig = _plot(
            df, x="PC", y="Variance Ratio",
            hover_data={"Variance Ratio": ":.2f"},
            labels={"x": f"PC", "y": f"Variance Ratio"},
            markers=True,
            # labels={"value": y_var_label, "Component": "PC"}
        )
        # fig.update_layout(yaxis_range=[-1, self.hist_n_pcs + 1])

    fig.update_layout(layout)
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="PC",
        yaxis_title="Variance Ratio",
        margin=dict(t=10, b=10, l=10, r=10),
    )
    fig.update_traces(
        hovertemplate="var: %{y:.2f}"
    )

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig


def pca_explain_corr(adata, y_var: Literal["R", "R^2"] = "R", n_pcs=None, layout=default_layout, fig_path=None):
    if n_pcs is None:
        n_pcs = adata.uns["pca"]["variance_ratio"].shape[0]

    cats = tools.get_categoric(adata)
    Rs = np.zeros((n_pcs, len(cats)))

    y_var_label = "R<sup>2</sup>" if y_var == "R^2" else "R"

    for i, cat in enumerate(cats):
        for j in range(n_pcs):
            Rs[j, i] = np.corrcoef(adata.obs[cat].cat.codes, adata.obsm["X_pca"][:, j])[0, 1]
            if y_var == "R^2":
                Rs[j, i] = Rs[j, i] ** 2

    df = pd.DataFrame(Rs, columns=cats, index=range(1, n_pcs+1))
    df = pd.melt(df, ignore_index=False).reset_index().rename(columns={"index": "Component", "variable": "Feature"})
    fig = px.scatter(
        df, x="Component", y="value", color="Feature",
        hover_data={"value": ":.2f", "Component": False},
        labels={"value": y_var_label, "Component": "PC"}
    )
    fig.update_layout(hovermode="x unified")
    fig.update_traces(
        hovertemplate=y_var_label + " = %{y:.2f}", marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey"))
    )
    fig.update_layout(layout)
    fig.update_layout(
        xaxis=dict(
            title="PC", tick0=1, dtick=1, showgrid=False, zeroline=False,
            visible=True, showticklabels=True,
        ),
        yaxis=dict(
            range=[0, 1], title="R<sup>2</sup>" if y_var == "R^2" else "R", tick0=0, dtick=0.2, showgrid=False,
            zeroline=False, visible=True, showticklabels=True,
        ),
        legend_title="Feature",
    )

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig


def pca_corr_circle(adata, components=(0, 1), layout=default_layout, fig_path=None):
    x = adata.varm["PCs"][:, components[0]]
    y = adata.varm["PCs"][:, components[1]]
    x_ratio = adata.uns["pca"]["variance_ratio"][components[0]]
    y_ratio = adata.uns["pca"]["variance_ratio"][components[1]]

    dist = np.sqrt(x**2 + y**2)

    text = adata.var_names.values.copy()
    text[dist < np.quantile(dist, 0.999)] = ""

    df = pd.DataFrame(
        {"x": x, "y": y, "Gene": adata.var_names.values, "text": text}
    )
    df["textposition"] = ""
    df.loc[df["y"] > 0, "textposition"] = (
        df.loc[df["y"] > 0, "textposition"] + "top"
    )
    df.loc[df["y"] == 0, "textposition"] = (
        df.loc[df["y"] == 0, "textposition"] + "center"
    )
    df.loc[df["y"] < 0, "textposition"] = (
        df.loc[df["y"] < 0, "textposition"] + "bottom"
    )

    df.loc[df["x"] < 0, "textposition"] = (
        df.loc[df["x"] < 0, "textposition"] + " left"
    )
    df.loc[df["x"] > 0, "textposition"] = (
        df.loc[df["x"] > 0, "textposition"] + " right"
    )

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xaxis_range = [xmin - 0.3 * xmax, xmax + 0.3 * xmax]
    yaxis_range = [ymin - 0.3 * ymax, ymax + 0.3 * ymax]

    fig = px.scatter(
        df, x="x", y="y", text="text", hover_name="Gene",
        hover_data={"x": ":.2f", "y": ":.2f", "Gene": False, "text": False},
        labels={"x": f"PC {components[0]+1}", "y": f"PC {components[1]+1}"}
    )

    fig.update_layout(layout)

    fig.update_traces(
        textposition=df["textposition"],
        marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey"))
    )

    fig.update_layout(
        xaxis_title=f"PC {components[0]+1} ({x_ratio*100:.1f} %)",
        yaxis_title=f"PC {components[1]+1} ({y_ratio*100:.1f} %)",
        xaxis_range=xaxis_range, yaxis_range=yaxis_range,
    )

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig
