"""Reusable Plotly helpers for jbubble visualisations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from .simulation import SimulationResult


@dataclass
class PlotArrays:
    """Convenient numpy arrays for plotting a simulation."""

    time_us: np.ndarray
    radius_um: np.ndarray
    pressure_kpa: np.ndarray


def arrays_from_result(result: SimulationResult) -> PlotArrays:
    units = result.units
    return PlotArrays(
        time_us=np.asarray(result.ts) / units.T_scale,
        radius_um=np.asarray(result.radius) / units.L_scale,
        pressure_kpa=np.asarray(result.driving_pressure) / units.P_scale,
    )


ArrayLike = Union[Sequence[float], np.ndarray]


def line_trace(x: ArrayLike, y: ArrayLike, *, name: str, color: str | None = None) -> go.Scatter:
    return go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color) if color else None)


def line_figure(
    traces: Iterable[go.Scatter],
    *,
    x_label: str,
    y_label: str,
    template: str = "plotly_white",
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    fig = go.Figure(data=list(traces))
    fig.update_layout(
        template=template,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig



__all__ = [
    "PlotArrays",
    "arrays_from_result",
    "line_trace",
    "line_figure",
]
