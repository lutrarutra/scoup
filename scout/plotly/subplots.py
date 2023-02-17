import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from ..tools import subplot_dims


def subplots(figs, ncols=None, nrows=None, width=500, height=500, fig_path=None, subplot_titles=None):
    n_figs = len(figs)

    ncols, nrows = subplot_dims(n_figs, ncols=ncols, nrows=nrows)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles)

    for i, _fig in enumerate(figs):
        row = int(np.floor(i / ncols)) + 1
        col = i % ncols + 1
        for trace in _fig.data:
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(width=width * ncols + 100, height=height * nrows, margin=dict(t=30, b=5, l=5, r=5))
    

    if fig_path is not None:
        fig.write_image(fig_path, scale=5)

    return fig

