### Custom functions to plot SC RNA-seq data
import os

import pandas as pd
import scanpy as sc
import numpy as np

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mline

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from .tools import subplot_dims, subplot_idx


def dispersion_plot(
    adata, groupby, color="abs_score", style=None,
    nrows=None, ncols=None,
    cmap="viridis", size=None,
    path=None, figsize=(6, 5), dpi=80,
    xlim=None, ylim=None, vmax=5.0
):
    n_figs = len(adata.obs[groupby].cat.categories)
    ncols, nrows = subplot_dims(n_figs, ncols=ncols, nrows=nrows)

    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=dpi)

    vmin = 0

    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=(vmax-vmin)/2, vmax=vmax)

    for i, group in enumerate(adata.obs[groupby].cat.categories):
        idx = subplot_idx(i, ncols=ncols, nrows=nrows)

        sns.scatterplot(
            x=adata.varm[f"mu_expression_{groupby}"][:, i],
            y=adata.varm[f"cv_{groupby}"][:, i] ** 2,
            c=np.log1p(adata.uns["de"][groupby][f"{group} vs. rest"][color].values),
            style=adata.var[style] if style is not None else None,
            size=size,
            edgecolor=(0, 0, 0, 1),
            # color=(0, 0, 0, 0),
            linewidth=0.8,
            ax=ax[idx],
            cmap=cmap, norm=norm
        )
        ax[idx].set_title(f"{group}")

        ax[idx].set_yscale("log")
        ax[idx].set_xscale("log")

        if xlim != None:
            ax[idx].set_xlim(xlim)
        if ylim != None:
            ax[idx].set_ylim(ylim)

    f.text(0.5, 0.1, "Mu Expression", ha="center")
    f.text(0.12, 0.5, "(CV)^2", va="center", rotation=90)

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.35, 0.03, 0.3])
    colorbar = f.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical", cax=cbar_ax
    )
    # colorbar.ax.set_ylabel(, fontsize=10, loc="center", labelpad=-40)

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()

def pseudo_bulk_plot(
    adata, style=None, figsize=(6, 5), dpi=80, ncols=None, nrows=None,
    cmap="seismic", vmin=None, vmax=None, vcenter=0.0, path=None
):
    n_figs = adata.varm["bulk"].shape[1]
    ncols, nrows = subplot_dims(n_figs, ncols=ncols, nrows=nrows)

    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=dpi)

    if vmin == None:
        vmin = adata.varm["pseudo_factor"].min()
    if vmax == None:
        vmax = adata.varm["pseudo_factor"].max()
    if vcenter == None:
        vcenter = 0.0

    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    if style is not None:
        if style in adata.var.columns:
            style = adata.var[style].values
        else:
            style = adata.varm[style]

    for i in range(len(adata.uns["bulk_samples"])):
        idx = subplot_idx(i, ncols=ncols, nrows=nrows)
        _ax = ax[idx] if type(ax) is np.ndarray else ax
        sns.scatterplot(
            x=adata.varm["pseudo"][:, 0],
            y=adata.varm["bulk"][:, i],
            c=adata.varm["pseudo_factor"][:, i],
            style=style[:, i] if style.ndim == 2 else style,
            edgecolor=(0, 0, 0, 1),
            ax=_ax,
            linewidth=1,
            cmap=cmap,
            norm=norm,
        ).set_title(f"Sample {adata.uns['bulk_samples'][i]}")

        _ax.plot(
            [0, max(adata.varm["pseudo"][:, 0].max(), adata.varm["bulk"][:, i].max())],
            [0, max(adata.varm["pseudo"][:, 0].max(), adata.varm["bulk"][:, i].max())],
            label="y=x",
            c="royalblue",
        )

        _ax.set_xscale("log")
        _ax.set_yscale("log")

        # scalarmappaple = matplotlib.cm.ScalarMappable(norm=normalize, cmap=colormap)
        # scalarmappaple.set_array(adata.varm["pseudo"][:, i])
        # f.colorbar(scalarmappaple, fraction=0.05, pad=0.01, shrink=0.5)

        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    f.text(0.5, 0.1, "Pseudo Bulk", ha="center")
    f.text(0.12, 0.5, "Bulk", va="center", rotation=90)

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.35, 0.03, 0.3])
    f.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical", cax=cbar_ax
    )

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()


def dropout_plot(
    adata, groupby, style=None,
    nrows=None, ncols=None,
    cmap="viridis", size=None,
    path=None, figsize=(6, 5), dpi=80,
    xlim=None, ylim=(-0.1, 1.1),
):
    n_figs = len(adata.obs[groupby].cat.categories)
    ncols, nrows = subplot_dims(n_figs, ncols=ncols, nrows=nrows)

    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=dpi)

    vmin = adata.varm[f"dropout_weight_{groupby}"].min()
    vmax = adata.varm[f"dropout_weight_{groupby}"].max()

    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=(vmax-vmin)/2, vmax=vmax)

    if style is not None:
        if style in adata.var.columns:
            style = adata.var[style].values
        else:
            style = adata.varm[style]

    for i, group in enumerate(adata.obs[groupby].cat.categories):
        idx = subplot_idx(i, ncols=ncols, nrows=nrows)

        sns.scatterplot(
            x=adata.varm[f"nan_log_mu_expression_{groupby}"][:, i],
            y=adata.varm[f"dropout_{groupby}"][:, i],
            c=adata.varm[f"dropout_weight_{groupby}"][:, i],
            style=style[:, i] if style.ndim == 2 else style,
            size=size,
            edgecolor=(0, 0, 0, 1),
            # color=(0, 0, 0, 0),
            linewidth=0.8,
            ax=ax[idx],
            cmap=cmap, norm=norm
        )
        ax[idx].set_title(f"{group}")
        if xlim != None:
            ax[idx].set_xlim(xlim)
        if ylim != None:
            ax[idx].set_ylim(ylim)

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.35, 0.03, 0.3])
    colorbar = f.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical", cax=cbar_ax
    )
    # colorbar.ax.set_ylabel(, fontsize=10, loc="center", labelpad=-40)

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()

def marker_volcano(df, x="logFC", y="-log_pvals_adj", hue="log_mu_expression", cmap="rocket", significance_threshold=0.05, fig_path=None, fig_size=(10, 8), dpi=80):
    f, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    clrs = df[hue]
    norm = matplotlib.colors.TwoSlopeNorm(vmin=clrs.min(), vcenter=(clrs.max()-clrs.min())/2, vmax=clrs.max())

    sns.scatterplot(
        data=df, x="logFC", y="-log_pvals_adj", c=df[hue], edgecolor=(0,0,0,1),
        ax=ax, linewidth=1, legend=False, norm=norm, cmap=cmap
    )

    show_genes = np.concatenate([
        df[df["logFC"] > 0].sort_values("-log_pvals_adj", ascending=False).index[:5],
        df[df["logFC"] < 0].sort_values("-log_pvals_adj", ascending=False).index[:5]
    ])

    xscale = np.abs(np.array(ax.get_xlim())).sum()
    yscale = np.abs(np.array(ax.get_ylim())).sum()

    for gene, row in df.iterrows():
        if gene in show_genes:
            x = row["logFC"]
            y = row["-log_pvals_adj"]
            if x > 0:
                xytext = (x + 0.03 * xscale, y + 0.03 * yscale)
            else:
                xytext = (x - 0.1 * xscale, y + 0.03 * yscale)

            ax.annotate(gene, xy=(x,y), fontsize=8, xytext=xytext, va="center", arrowprops=dict(arrowstyle="-"))

    ax.axhline(-np.log10(0.05), linestyle="--", color="royalblue", linewidth=1)

    colorbar = f.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical", fraction=0.1, pad=0.03, shrink=0.5, 
    )
    colorbar.ax.set_ylabel(hue.replace("_", " ").title(), fontsize=10, loc="center", labelpad=-40)

    true_marker = mline.Line2D(
        [], [], markerfacecolor=sns.color_palette(cmap)[-1], marker='o', linestyle='None',
        markersize=7, label='True', markeredgewidth=1, markeredgecolor="black"
    )
    false_marker = mline.Line2D(
        [], [], markerfacecolor=sns.color_palette(cmap)[-1], marker='X', linestyle='None',
        markersize=7, label='False', markeredgewidth=1, markeredgecolor="black"
    )

    legend = ax.legend(title=f"Significant (p-value < {significance_threshold})", handles=[true_marker, false_marker], prop={'size': 8})
    plt.setp(legend.get_title(),fontsize='x-small')
    ax.set_ylabel("- log adjusted p-value")

    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")

    plt.show()

def plot_marker_score_distributions(df, fig_path=None, fig_size=(12, 6), dpi=80):
    f, ax = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)
    sns.histplot(data=df, x="pvals_adj", ax=ax[0], bins=20)
    sns.histplot(data=df, x="gene_score", ax=ax[1], bins=50)
    # ax[1].set_yscale("log")
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")
    plt.show()


def plot_marker_scores(df, x="logFC", y="gene_score", hue="log_mu_expression", cmap="rocket", significance_threshold=0.05, fig_path=None, fig_size=(12, 6), dpi=80):
    f, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
    clrs = df[hue]
    norm = matplotlib.colors.TwoSlopeNorm(vmin=clrs.min(), vcenter=(clrs.max()-clrs.min())/2, vmax=clrs.max())

    sns.scatterplot(
        data=df, x=x, y=y, c=clrs, style=df["pvals_adj"] < significance_threshold, legend=False,
        edgecolor=(0,0,0,1), ax=ax, linewidth=1, style_order=[True, False], norm=norm, cmap=cmap

    )
    colorbar = f.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical", fraction=0.1, pad=0.03, shrink=0.5, 
    )
    colorbar.ax.set_ylabel(hue.replace("_", " ").title(), fontsize=10, loc="center", labelpad=-40)

    true_marker = mline.Line2D(
        [], [], markerfacecolor=sns.color_palette(cmap)[-1], marker='o', linestyle='None',
        markersize=7, label='True', markeredgewidth=1, markeredgecolor="black"
    )
    false_marker = mline.Line2D(
        [], [], markerfacecolor=sns.color_palette(cmap)[-1], marker='X', linestyle='None',
        markersize=7, label='False', markeredgewidth=1, markeredgecolor="black"
    )

    legend = ax.legend(title=f"Significant (FDR < {significance_threshold})", handles=[true_marker, false_marker], prop={'size': 8})
    plt.setp(legend.get_title(),fontsize='x-small')

    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")

    plt.show()

def gsea_volcano(gsea_df, x="nes", y="-log10_fdr", hue="matched_fraction", cmap="rocket", significance_threshold=0.05, fig_path=None, fig_size=(12, 6), dpi=80):
    rcParams["axes.grid"] = False
    f, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    norm = matplotlib.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    sns.scatterplot(
        data=gsea_df, x=x, y=y, c=gsea_df[hue], style=gsea_df["fdr"] < significance_threshold,
        legend=False, norm=norm, edgecolor=(0,0,0,1), ax=ax, linewidth=1, style_order=[True, False], cmap=cmap
    )

    show_terms = np.concatenate([
        gsea_df.sort_values("fdr", ascending=True).index[:5],
    ])

    xscale = np.abs(np.array(ax.get_xlim())).sum()
    yscale = np.abs(np.array(ax.get_ylim())).sum()

    for idx, row in gsea_df.iterrows():
        if idx in show_terms:
            # ax.text(row["logFC"]+0.2, row["-log_pvals_adj"]+0.2, gene, fontsize=8)
            x = row["nes"]
            y = row["-log10_fdr"]
            if x > 0:
                xytext = (x - 0.1 * xscale, y + 0.03 * yscale)
            else:
                xytext = (x + 0.03 * xscale, y + 0.0 * yscale)

            ax.annotate(row["Term"], xy=(x,y), fontsize=9, xytext=xytext, va="center")


    true_marker = mline.Line2D(
        [], [], markerfacecolor=sns.color_palette(cmap)[-1], marker='o', linestyle='None',
        markersize=7, label='True', markeredgewidth=1, markeredgecolor="black"
    )
    false_marker = mline.Line2D(
        [], [], markerfacecolor=sns.color_palette(cmap)[-1], marker='X', linestyle='None',
        markersize=7, label='False', markeredgewidth=1, markeredgecolor="black"
    )

    colorbar = f.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical", fraction=0.1, pad=0.03, shrink=0.5, 
    )
    colorbar.ax.set_ylabel(hue.replace("_", " ").title(), fontsize=10, loc="center", labelpad=-50)

    legend = ax.legend(title=f"Significant (FDR < {significance_threshold})", handles=[true_marker, false_marker], prop={'size': 8})
    plt.setp(legend.get_title(),fontsize='x-small')

    ax.axhline(-np.log10(significance_threshold), linestyle="--", color="royalblue", linewidth=1)
    ax.set_ylabel("-Log 10 FDR")
    ax.set_xlabel("Normalised Enrichment Score (NES)")

    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")

    plt.show()


def _dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count        # axes[0][i+1].title.set_text(f"{clusterby.capitalize()} {groups[i]}")

    linkage_matrix = np.column_stack(
        [model.children_, np.linspace(0, 1, model.distances_.shape[0]+2)[1:-1], counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)


def heatmap(
    adata, groupby, categorical_features, var_names=None, rank_genes_by=None, free_sort_cells=False,
    n_genes=10, sort_cells=True, sort_genes=True, quantiles=(0.0, 1.0), cmap="seismic", figsize=(20, None), dpi=50, fig_path=None):
    if isinstance(categorical_features, str):   
        categorical_features = [categorical_features]
    
    _grid = rcParams["axes.grid"]
    rcParams["axes.grid"] = False


    palettes = [adata.uns[f"{var}_colors"] if f"{var}_colors" in adata.uns.keys() else sns.color_palette("Paired") for var in categorical_features]
    na_clr = matplotlib.colors.cnames["lightgrey"]

    if var_names is None:
        rank_results = sc.tl.rank_genes_groups(
            adata, groupby=rank_genes_by if rank_genes_by else groupby,
            rankby_abs=True, method="t-test", copy=True
        ).uns["rank_genes_groups"]
        var_names = np.unique(np.array(list(map(list, zip(*rank_results["names"]))))[:,:n_genes].flatten())
    else:
        if isinstance(var_names, list):
            var_names = np.array(var_names)

    n_cat = len(categorical_features)
    h_cat = 0.5
    n_vars = len(var_names)
    h_vars = 0.3

    if figsize[1] is None:
        figsize = (figsize[0], n_cat * h_cat + n_vars * h_vars)

    r_cat = int(h_cat * 100.0 / (n_cat * h_cat + n_vars * h_vars))
    r_vars = 100 - r_cat

    f = plt.figure(figsize=figsize, dpi=dpi)

    gs = f.add_gridspec(
        1 + len(categorical_features), 2, hspace=0.2/figsize[1], wspace=0.01,
        height_ratios=[r_cat] * len(categorical_features) + [r_vars], width_ratios=[1, 20]
    )
    
    axes = gs.subplots(sharex="col", sharey="row")

    if sort_genes:
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(adata[:,var_names].X.T.toarray())
        gene_dendro = _dendrogram(model, ax=axes[-1, 0], orientation="right")
        gene_order = var_names[gene_dendro["leaves"]]

        icoord = np.array(gene_dendro["icoord"])
        icoord = icoord / (n_genes*10) * n_genes
        dcoord = np.array(gene_dendro["dcoord"])
        axes[-1, 0].clear()
        for xs, ys in zip(icoord, dcoord):
            axes[-1, 0].plot(ys, xs)
    else:
        gene_order = var_names

    if sort_cells:
        adata.obs["barcode"] = pd.Categorical(adata.obs.index)

        if free_sort_cells:
            if not f"dendrogram_barcode" in adata.uns.keys():
                sc.tl.dendrogram(adata, groupby="barcode", var_names=var_names)
            cell_dendro = adata.uns["dendrogram_barcode"]
            cell_order = cell_dendro["categories_ordered"]
        else:
            cell_order = []
            for cell_type in adata.obs[groupby].cat.categories.tolist():
                # TODO: when reculcustering, the order of the cells is not preserved
                # if f"{cell_type}_order" in adata.uns.keys():
                #     _dendro = list(set(adata.uns[f"{cell_type}_order"]) & set(adata.obs_names))
                #     cell_order.extend(_dendro)
                #     cell_order.extend(_dendro)
                # else:
                # TODO: multithreading
                dendro = sc.tl.dendrogram(adata[adata.obs[groupby] == cell_type], groupby="barcode", inplace=False)
                _dendro = list(set(dendro["categories_ordered"]) & set(adata.obs_names))
                cell_order.extend(_dendro)
                # adata.uns[f"{cell_type}_order"] = dendro["categories_ordered"]

    else:
        cell_order = adata.obs.sort_values(groupby).index

    data = adata[cell_order, gene_order].layers["logcentered"].toarray().T
    vmin, vmax = np.quantile(data, q=quantiles)

    sns.heatmap(
        data, cmap=cmap, ax=axes[-1,-1],
        center=0, vmin=vmin, vmax=vmax, cbar=False, yticklabels=gene_order,
    )

    for i, categorical_feature in enumerate(categorical_features):
        palette = palettes[i % len(palettes)]
        samples = adata[cell_order, :].obs[categorical_feature].cat.codes
        clr = [palette[s % len(palette)] if s != -1 else na_clr for s in samples]
        axes[i][1].vlines(np.arange(len(samples)), 0, 1, colors=clr, lw=5, zorder=10)
        axes[i][1].set_yticklabels([])
        axes[i][1].set_ylim([0,1])
        axes[i][1].patch.set_linewidth(2.0)
        axes[i][1].set_yticks([0.5])
        axes[i][1].set_yticklabels([categorical_feature.capitalize()])

        leg = f.legend(
            title=categorical_feature, labels=adata.obs[categorical_feature].cat.categories.tolist(),
            prop={"size": 24}, bbox_to_anchor=(0.95, 0.9 - 0.3*i), ncol=1, frameon=True, edgecolor="black",
            loc="upper left", facecolor="white"
        )

        plt.gca().add_artist(leg)
        palette = palettes[i%len(palettes)]
        for l, legobj in enumerate(leg.legendHandles):
            legobj.set_color(palette[l % len(palette)])
            legobj.set_linewidth(8.0)

    for ax in axes.flat:
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.set_xticklabels([])
    
    for ax in axes[:,0]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    f.colorbar(
        plt.cm.ScalarMappable(norm=matplotlib.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0), cmap=cmap),
        ax=axes, orientation="vertical", fraction=0.05, pad=0.01, shrink=5.0 / figsize[1]
    )

    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")

    plt.show()
    rcParams["axes.grid"] = _grid

def clustermap(adata, clusterby="leiden", categorical_features="sample", layer="logcentered", n_genes=10, figsize=(25, 25), dpi=50, fig_path=None, min_cells=100):
    
    # Make barcode into a categorical column so that we can dendrogram by cells
    adata.obs["barcode"] = pd.Categorical(adata.obs.index)

    if type(categorical_features) != list:
        categorical_features = [categorical_features]
    
    palettes = [adata.uns[f"{var}_colors"] if f"{var}_colors" in adata.uns.keys() else sns.color_palette("Paired") for var in categorical_features]
    na_clr = matplotlib.colors.cnames["lightgrey"]

    rcParams["axes.grid"] = False

    # Filter groups with more than 'min_cells' cells
    cluster_sz = adata.obs.groupby(clusterby).apply(len)
    cluster_sz = cluster_sz[cluster_sz >= min_cells]
    groups = cluster_sz.index.tolist()
    n_groups = len(groups)
    n_cells = min(cluster_sz)
    print(f"Using {n_cells} cells from {n_groups} {clusterby}s")

    # Rank genes and make 2d array
    sc.pp.neighbors(adata, random_state=0)
    rank_results = sc.tl.rank_genes_groups(adata, groupby=clusterby, method="t-test", copy=True).uns["rank_genes_groups"]
    gene_groups = np.array(list(map(list, zip(*rank_results["names"]))))[:,:n_genes]

    f = plt.figure(figsize=figsize, dpi=dpi)
    gs = f.add_gridspec(n_groups+len(categorical_features), n_groups+1, hspace=0.05, wspace=0.05, height_ratios=[1] * len(categorical_features) + [10] * n_groups, width_ratios=[2] + [10] * n_groups)
    axes = gs.subplots(sharex="col", sharey="row")

    # Color bar minimum and maximum
    vmin = adata.layers[layer].min()
    vmax = adata.layers[layer].max()

    # Category label color plot
    for gi, grp in enumerate(categorical_features):
        axes[gi][0].set_yticks([0.5])
        axes[gi][0].set_yticklabels([grp.capitalize()])
        adata.obs[f"{grp}_idx"] = adata.obs[grp].cat.codes

    for i, group_i in enumerate(groups):
        # Index n_cells per cluster and top n_genes
        sample_idx = adata.obs[adata.obs[clusterby] == group_i].sample(n_cells).index
        adata_sample = adata[sample_idx, :].copy()

        dendro_info = sc.tl.dendrogram(adata_sample, groupby="barcode", var_names=gene_groups.flatten(), inplace=False)
        dendro_order = dendro_info["categories_ordered"]

        # Gene hierachy
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(adata[:,gene_groups[i,:]].X.T.toarray())
        axes[i+len(categorical_features)][0].set_ylim([0,1])
        gene_dendro = _dendrogram(model, ax=axes[i+len(categorical_features), 0], orientation="right")
        gene_groups[i, :] = gene_groups[i, gene_dendro["leaves"]]
        icoord = np.array(gene_dendro["icoord"])
        icoord = icoord / (n_genes*10) * n_genes
        dcoord = np.array(gene_dendro["dcoord"])

        axes[i+len(categorical_features), 0].clear()
        for xs, ys in zip(icoord, dcoord):
            axes[i+len(categorical_features), 0].plot(ys, xs)

        # Category label color plot
        for gi, grp in enumerate(categorical_features):
            palette = palettes[gi%len(palettes)]
            samples = adata_sample[dendro_order,:].obs[f"{grp}_idx"].values
            clr = [palette[i % len(palette)] if i != -1 else na_clr for i in samples]
            axes[gi][i+1].vlines(np.arange(len(samples)), 0, 1, colors=clr, lw=5, zorder=10)
            axes[gi][i+1].patch.set_edgecolor("black")
            axes[gi][i+1].patch.set_linewidth(2.0)

        axes[0][i+1].set_title(f"{clusterby.capitalize()} {groups[i]}" if clusterby == "leiden" else groups[i], fontsize=18)
        axes[i+len(categorical_features)][0].set_ylabel(f"{clusterby.capitalize()} {groups[i]}" if clusterby == "leiden" else groups[i], fontsize=18)

        for j, group_j in enumerate(groups):
            data = adata[dendro_order, gene_groups[j,:]].layers[layer].T.toarray()
            sns.heatmap(
                data,
                ax=axes[j+len(categorical_features)][i+1], cmap="seismic", center=0, vmin=vmin, vmax=vmax, cbar=False, yticklabels=gene_groups[j,:]
            )
            axes[j+len(categorical_features)][i+1].patch.set_edgecolor("black")
            axes[j+len(categorical_features)][i+1].patch.set_linewidth(2.0)
        
    f.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0), cmap="seismic"), ax=axes, orientation="vertical", fraction=0.05, pad=0.05, shrink=0.2)

    for gi, grp in enumerate(categorical_features):
        leg = f.legend(title=grp, labels=adata.obs[grp].cat.categories.tolist(), prop={"size": 24}, bbox_to_anchor=(0.95, 1.0 - 0.1 - gi/12))
        plt.gca().add_artist(leg)
        palette = palettes[gi%len(palettes)]
        for l, legobj in enumerate(leg.legendHandles):
            legobj.set_color(palette[l % len(palette)])
            legobj.set_linewidth(8.0)

    # Hide frames from dendrogram column
    for i in range(axes.shape[0]):
        axes[i][0].spines['top'].set_visible(False)
        axes[i][0].spines['right'].set_visible(False)
        axes[i][0].spines['bottom'].set_visible(False)
        axes[i][0].spines['left'].set_visible(False)

    for ax in axes.flat:
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.set_xticklabels([])

    f.align_ylabels(axes[:, 0])

    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")

    plt.show()

##########################################################################################


#     OLD PLOTS


##########################################################################################


def scatter_pca(adata: sc.AnnData, x:int, y:int, hue=None, style=None, add_labels=False, label_col=None, fig_path=None, fig_size=(10,8)):
    rcParams['figure.figsize'] = fig_size
    sns.set_theme(style="darkgrid")

    if not "X_pca" in adata.obsm.keys():
        sc.pp.pca(adata, n_comps=max(x+1,y+1))
    elif adata.obsm["X_pca"].shape[1] < max(x+1, y+1):
        sc.pp.pca(adata, n_comps=max(x+1,y+1))

    df = pd.concat([pd.DataFrame(adata.obsm["X_pca"]), adata.obs.reset_index(drop=True)], axis="columns")
    p1 = sns.scatterplot(x=x, y=y, data=df, hue=hue, style=style, s=100)

    if add_labels:
        assert label_col != None
        for line in range(0,adata.shape[0]):
            p1.text(df[x][line]-7, df[y][line], 
                    df[label_col][line], verticalalignment="center", horizontalalignment='center', 
                    size='x-small', color='black', weight='normal', alpha=1.0)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")
        
    plt.xlabel(f"PC {x+1} ({adata.uns['pca']['variance_ratio'][x]*100.0:.1f} %)")
    plt.ylabel(f"PC {y+1} ({adata.uns['pca']['variance_ratio'][y]*100.0:.1f} %)")

    
    plt.show()

def explain_variance(adata):
    f, ax = plt.subplots(ncols=2, figsize=(18, 6))
    n_components = len(adata.uns['pca']['variance_ratio'])
    ax[0].bar(range(1,n_components+1), adata.uns['pca']['variance_ratio'])
    ax[1].plot(range(1,n_components+1), adata.uns['pca']['variance_ratio'].cumsum())
    ax[0].title.set_text("Var explained")
    ax[1].title.set_text("Cumulative")
    ax[0].set_xticks(range(1,n_components+1))
    ax[1].set_xticks(range(1,n_components+1))

    ax[0].set_xlabel(f"Principal Component")
    ax[0].set_ylabel(f"Ratio")

    ax[1].set_xlabel(f"Principal Component")
    ax[1].set_ylabel(f"Ratio")

    plt.show()


def scatter_check(adata: sc.AnnData, x_type: str, y_type:str, group_by="sample", fig_path=None, fig_size=(10, 6), return_residuals=False):
    groups = adata.obs[group_by].unique()
    nrows = int(np.ceil(len(groups)/2))

    f, ax = plt.subplots(nrows, 2, figsize=(fig_size[0], fig_size[1]*nrows))
    residuals = np.empty((len(groups), adata.shape[1]))

    for i, group in enumerate(groups):
        fj = i % 2
        fi = int(i / 2)
        x = adata[(adata.obs[group_by]==group) & (adata.obs["type"]==x_type)].X[0]
        y = adata[(adata.obs[group_by]==group) & (adata.obs["type"]==y_type)].X[0]
        residuals[i] = x - y
        ax[fi][fj].set_title(f"Pearson: {np.corrcoef(x,y)[0][1]:.2f}")
        ax[fi][fj].set_xlabel(f"{group}_{x_type}")
        ax[fi][fj].set_ylabel(f"{group}_{y_type}")
        ax[fi][fj].scatter(x=x, y=y, s=0.1)
        ax[fi][fj].plot([0,10], [0,10], label="y=x", c="red")
        ax[fi][fj].legend()
    
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight")
    
    plt.show()
    if return_residuals:
        return pd.DataFrame(data=residuals.T, index=adata.var.index, columns=groups)


def plot_corr(corr, fig_path=None, print_index=True):
    f, ax = plt.subplots(figsize=(12, 8))
    im = ax.matshow(corr, cmap="viridis")

    for i, row in enumerate(corr.values):
        z = np.argsort(np.argsort(row))
        j_max = np.argmax(row)
        for j, col in enumerate(row):
            if print_index:
                ax.text(j,i, f"{len(row) - z[j]}",
                    ha="center", va="center",
                    fontsize=8, color="red", fontweight="light"
                )
            else:
                ax.text(j,i, f"{corr.values[i][j]:.2f}",
                    ha="center", va="center",
                    fontsize=8, color="red", fontweight="extra bold" if j_max==j else "light"
                )

    plt.grid(False)
    plt.xticks(range(corr.columns.values.shape[0]), corr.columns.values, rotation=90)
    plt.yticks(range(corr.index.values.shape[0]), corr.index.values)

    cb = plt.colorbar(im)

    if fig_path:
        plt.savefig(fig_path, bbox_inches = "tight")
        
    plt.show()


def bar_proportions(proportions, figsize=(8,8), dpi=100, fig_path=None, palette="tab10", xlabels_rotation=0, bar_label_fontsize=12):
    palette=sns.color_palette(palette, n_colors=len(proportions.columns))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    proportions.plot(kind="bar", stacked=True, ax=ax, color=palette, width=0.7, alpha=1.0)

    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=proportions.columns)
    ax.grid(False)
    
    # return ax.containers
    for bars in ax.containers:
        labels = []
        for val in bars.datavalues:
            if val > 0.03:
                labels.append(f"{val*100:.0f}%")
            else:
                labels.append("")
        ax.bar_label(bars, padding=-16, labels=labels, fontsize=bar_label_fontsize)

    plt.xticks(rotation=xlabels_rotation)
    
    if fig_path:
        plt.savefig(fig_path, bbox_inches = "tight")


    plt.show()


def compare_proportions(p_true, p_est, fig_path=None):
    palette=sns.color_palette("hls", n_colors=max(len(p_true.columns), len(p_est.columns)))

    t_true = pd.DataFrame(p_true.values, index=p_true.index + "_true", columns=p_true.columns)
    t_est = pd.DataFrame(p_est.values, index=p_est.index + "_est", columns=p_est.columns)
    df = pd.concat([t_true, t_est]).sort_index()

    fig, ax = plt.subplots(figsize=(18,13))


    for i in range(0, len(p_est.columns)*2, 2):
        if i % 4 == 0:
            plt.axvspan(i-0.5, i+1.5, facecolor="grey", alpha=0.5)

    df.plot(kind="bar", stacked=True, ax=ax, color=palette, width=0.7, alpha=1.0)
    ax.grid(False)

    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    
    plt.ylabel("Proportion")

    if fig_path:
        plt.savefig(fig_path, bbox_inches = "tight")

    for bars in ax.containers:
        labels = []
        for val in bars.datavalues:
            if val > 0.03:
                labels.append(f"{val*100:.0f}%")
            else:
                labels.append("")
        ax.bar_label(bars, padding=-16, labels=labels)

    plt.show()


def scatter_proportions(p_true, p_est, fig_path=None):
    palette=sns.color_palette("hls", n_colors=max(len(p_true.columns), len(p_est.columns)))
    
    groups = p_true.index.values

    fig, ax = plt.subplots(figsize=(10,10))

    df = pd.DataFrame(columns=["sample", "cluster", "true", "estimate"])

    for i, sample in enumerate(p_true.index):
        for j, cluster in enumerate(p_true.columns):
            df.loc[i * len(p_true.index) + j] = [sample, cluster, p_true.loc[sample, cluster], p_est.loc[sample, cluster]]

    ax = sns.scatterplot(x="true", y="estimate", style="sample", hue="cluster", data=df, s=200)
    ax.plot([0,1], [0,1], label="y=x")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    if fig_path:
        plt.savefig(fig_path, bbox_inches = "tight")

    plt.show()