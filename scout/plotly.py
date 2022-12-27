from typing import Literal

import chart_studio.plotly as py
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scanpy as sc
from plotly.subplots import make_subplots
import scipy

from . import tools as tools

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

PLOTLY_DISCRETE_COLORS = px.colors.qualitative.Plotly
SC_DEFAULT_COLORS = sc.pl.palettes.default_20
NONE_COLOR = "#d3d3d3"

_discrete_cmap_mapping = {
    "scanpy_default": SC_DEFAULT_COLORS,
    "plotly_qualitative": PLOTLY_DISCRETE_COLORS,
}

for obj in dir(px.colors.qualitative):
    if not obj.startswith("_"):
        if type(getattr(px.colors.qualitative, obj)) == list:
            _discrete_cmap_mapping[f"plotly_{obj}"] = getattr(px.colors.qualitative, obj)


def seismic(zcenter, wcenter=0.01):
    return [
        (0, "#00004C"),
        (zcenter * 0.5, "#0000E6"),
        (zcenter - zcenter * wcenter, "white"),
        (zcenter, "white"),
        (zcenter + zcenter * wcenter, "white"),
        (1 - (zcenter * 0.5), "#FF0808"),
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
        hover_data={"Bin": borders, "Proportion": proportions},
    )
    fig.update_layout(layout)
    fig.update_layout(
        xaxis_title=x.replace("_", " ").title(), yaxis_title="Count", bargap=0
    )
    fig.update_traces(marker=dict(line=dict(color="black", width=1)))

    if fig_path is not None:
        fig.write_image(fig_path)

    return fig

def pca_explain_variance(
    adata, plot_type: Literal["Cumulative", "Bar", "Line", "Area"] = "Cumulative",
    n_pcs=None, layout=_layout, fig_path=None
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
        fig.write_image(fig_path)

    return fig

def pca_explain_corr(adata, y_var: Literal["R", "R^2"] = "R", n_pcs=None, layout=_layout, fig_path=None):
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
        fig.write_image(fig_path)

    return fig

def pca_correlation_circle(adata, components=(0,1), layout=_layout, fig_path=None):
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
        fig.write_image(fig_path)

    return fig


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
    adata, obsm_layer: str = "X_umap", hue=None, hue_layer="log1p", hue_aggregate: Literal["abs", None] = "abs",
    fig_path=None, continuous_cmap="viridis", discrete_cmap=sc.pl.palettes.default_20, layout=_layout, components=None,
):
    fig = go.Figure()
    if type(discrete_cmap) == str:
        discrete_cmap = _discrete_cmap_mapping[discrete_cmap]

    if hue is None:
        color = None
        hue_title = None
        cmap = None
    else:
        if hue in adata.obs_keys():
            hue_title = hue
            color = adata.obs[hue]
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
                    cmap = seismic(zcenter)
                else:
                    cmap = continuous_cmap

        else:
            if type(hue) == str:
                hue_title = hue
                if hue_layer == "log1p" or hue_layer == "X":
                    color = adata.X[:, adata.var.index.get_loc(hue)]
                elif hue_layer in adata.layers.keys():
                    color = adata.layers[hue_layer][:, adata.var.index.get_loc(hue)]
                else:
                    assert False
            elif type(hue) == list:
                hue_title = "Marker Score"
                if hue_aggregate == "abs":
                    color = np.abs(adata[:, hue].layers["logcentered"]).mean(1)
                elif hue_aggregate == None:
                    color = adata[:, hue].layers["logcentered"].mean(1)
            else:
                assert False

            if isinstance(color, scipy.sparse.csr_matrix):
                color = color.toarray()

            if continuous_cmap == "seismic":
                zmin, zmax = np.quantile(color, [0.0, 1.0])
                zcenter = abs(zmin) / (zmax - zmin)
                cmap = seismic(zcenter)
            else:
                cmap = continuous_cmap

            color = color.T

    axis_title = obsm_layer.replace("X_", "").replace("_", " ").upper()


    if (adata.obsm[obsm_layer].shape[1] == 2) or (components is not None and len(components) == 2):
        if components == None:
            components = (0, 1)

        scatter = px.scatter(
            x=adata.obsm[obsm_layer][:, components[0]],
            y=adata.obsm[obsm_layer][:, components[1]],
            color=color,
            color_discrete_sequence=cmap,   
            color_continuous_scale=cmap,
            labels={
                "x": f"{axis_title} 1",
                "y": f"{axis_title} 2",
                "color": hue_title
            },
        )
        scatter.update_traces(marker=dict(
            size=6, line=dict(color="black", width=1)
        ), showlegend=False)

        scatter.update_layout(showlegend=True)

        # fig = _add_traces(fig, scatter)
        fig = _add_traces(scatter, fig)
        fig.update_layout(
            xaxis_title=f"{axis_title} 1",
            yaxis_title=f"{axis_title} 2",
        )


    else:
        if components == None:
            components = (0, 1, 2)

        scatter = px.scatter_3d(
            x=adata.obsm[obsm_layer][:, components[0]],
            y=adata.obsm[obsm_layer][:, components[1]],
            z=adata.obsm[obsm_layer][:, components[2]],
            color=color,
            color_discrete_sequence=cmap,
            color_continuous_scale=cmap,
            labels={
                "x": f"{axis_title} 1",
                "y": f"{axis_title} 2",
                "z": f"{axis_title} 3",
                "color": hue_title
            },
        )

        scatter.update_traces(marker=dict(
            size=3, line=dict(color="black", width=1)
        ), showlegend=False)

        scatter.update_layout(showlegend=True)

        # fig = _add_traces(fig, scatter)
        fig = _add_traces(scatter, fig)
        fig.update_layout(
            scene=go.layout.Scene(
                xaxis_title=f"{axis_title} 1",
                yaxis_title=f"{axis_title} 2",
                zaxis_title=f"{axis_title} 3",
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


# Plotly
def violin(
    data, y, groupby=None, layer=None, scatter=True, box=True,
    mean_line=False, scatter_size=1.0, jitter=0.6,
    violin_colors=SC_DEFAULT_COLORS, layout=_layout, fig_path=None,
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
        fig.write_image(fig_path)

    return fig


def gsea_volcano(
    gsea_df, x="nes", y="-log10_fdr", hue="matched_fraction",
    cmap="rocket", significance_threshold=0.05, fig_path=None, layout=_layout):

    fig = px.scatter(
        gsea_df.reset_index(), x=x, y=y, color=hue, hover_name="Term",
        hover_data={x: ":.2f", y: ":.2f", hue: ":.2f"}, color_continuous_scale="Viridis"
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
        fig.write_image(fig_path)

    return fig

def marker_volcano(
    df, x="logFC", y="-log_pvals_adj", hue="log_mu_expression",
    significance_threshold=0.05, cmap="plasma", layout=_layout, fig_path=None):

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
        fig.write_image(fig_path)

    return fig


def _create_categorical_row(adata, category, cmap=SC_DEFAULT_COLORS):
    c = adata.obs[category].cat.codes.values
    cats = adata.obs[category].cat.categories.values
    n_cats = len(cats)
    color_scale = [cmap[i % len(cmap)] for i in range(n_cats)]

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
    adata, var_names, categoricals=None, layer="logcentered", fig_path=None,
    layout=_layout, cluster_cells_by=None, cmap=None, 
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
    cmaps = [SC_DEFAULT_COLORS, PLOTLY_DISCRETE_COLORS]

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
                    adata[adata.obs[cluster_cells_by] == cell_type, :], groupby="barcode",
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
            colorscale = seismic(zcenter)
        else:
            colorscale = "viridis"
    else:
        if cmap == "seismic":
            colorscale = seismic(zcenter)
        else:
            colorscale = "viridis"


    fig.add_trace(
        go.Heatmap(z=z.T, y=y_ticks, showlegend=False, colorscale=colorscale),
        row=len(categoricals) + 1, col=2,
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
            fig.update_layout(
                {
                    ax: dict(
                        showgrid=False,
                        zeroline=False,
                        visible=False,
                        showticklabels=False,
                        range=[0, adata.n_obs],
                    )
                }
            )
        if ax[:5] == "yaxis":
            fig.update_layout(
                {
                    ax: dict(
                        showgrid=False,
                        zeroline=False,
                        visible=True,
                        showticklabels=True,
                    )
                }
            )

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
        fig.write_image(fig_path)

    return fig
