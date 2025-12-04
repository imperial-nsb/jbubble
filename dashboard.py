"""Dash + Plotly dashboard for jbubble simulations."""

from __future__ import annotations

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from jbubble import Units, SaveSpec, build_pulse, compute_radius_metrics, default_bubble, run_simulation

UNITS = Units()
NUM_SAMPLES = 800
PULSE_OPTIONS = [
    {"label": "Sine", "value": "sine"},
    {"label": "Sawtooth", "value": "sawtooth"},
    {"label": "Triangle", "value": "triangle"},
    {"label": "Square", "value": "square"},
]


def simulate(shape: str, freq_khz: float, pressure_mpa: float, radius_um: float, use_hann: bool):
    bubble = default_bubble(R0=radius_um * 1e-6)
    pulse = build_pulse(
        shape,
        freq=freq_khz * 1e3,
        pressure=pressure_mpa * 1e6,
        cycle_num=10,
        initial_time=1e-6,
        apply_hann=use_hann,
    )
    result = run_simulation(
        bubble=bubble,
        pulse=pulse,
        units=UNITS,
        save_spec=SaveSpec(num_samples=NUM_SAMPLES),
    )
    metrics = compute_radius_metrics(result)
    ts_trace_us = np.array(result.ts) * 1e6
    radius_trace_um = np.array(result.radius) * 1e6
    pressure_trace_kpa = np.array(result.driving_pressure) / 1e3
    return result, ts_trace_us, radius_trace_um, pressure_trace_kpa, metrics


def build_line_figure(x, y, x_label, y_label, name):
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y, mode="lines", name=name)],
        layout=go.Layout(
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title=x_label,
            yaxis_title=y_label,
        ),
    )
    fig.update_layout(template="plotly_white")
    return fig


def bubble_snapshot(radius_um: float):
    theta = np.linspace(0, 2 * np.pi, 120)
    x = radius_um * np.cos(theta)
    y = radius_um * np.sin(theta)
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y, fill="toself", mode="lines", line=dict(color="#888"))],
        layout=go.Layout(
            xaxis=dict(scaleanchor="y", showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            margin=dict(l=20, r=20, t=20, b=20),
            shapes=[
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=-radius_um,
                    y0=-radius_um,
                    x1=radius_um,
                    y1=radius_um,
                    line=dict(color="#888"),
                )
            ],
        ),
    )
    fig.update_layout(template="plotly_white")
    return fig


def layout_defaults():
    return dict(shape="sine", freq=800.0, pressure=1.0, radius=4.0, hann=False)


def serve_layout():
    defaults = layout_defaults()
    return html.Div(
        [
            html.H2("jbubble"),
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Pulse parameters"),
                            dcc.Dropdown(
                                id="shape-dropdown",
                                options=PULSE_OPTIONS,
                                value=defaults["shape"],
                                clearable=False,
                            ),
                            dcc.Slider(
                                id="freq-slider",
                                min=100,
                                max=2000,
                                step=50,
                                value=defaults["freq"],
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Small("Frequency (kHz)"),
                            dcc.Slider(
                                id="pressure-slider",
                                min=0.1,
                                max=1.5,
                                step=0.05,
                                value=defaults["pressure"],
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Small("Pressure (MPa)"),
                            dcc.Checklist(
                                id="hann-check",
                                options=[{"label": "Hann window", "value": "hann"}],
                                value=["hann"] if defaults["hann"] else [],
                                style={"marginTop": "0.5rem"},
                            ),
                        ],
                        style={"flex": "1", "padding": "1rem", "border": "1px solid #ddd"},
                    ),
                    html.Div(
                        [
                            html.P("Bubble parameters"),
                            dcc.Slider(
                                id="radius-slider",
                                min=1.0,
                                max=10.0,
                                step=0.1,
                                value=defaults["radius"],
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Small("Equilibrium radius (μm)"),
                        ],
                        style={"flex": "1", "padding": "1rem", "border": "1px solid #ddd"},
                    ),
                ],
                style={"display": "flex", "gap": "1rem", "marginBottom": "1rem"},
            ),
            html.Div(
                [
                    dcc.Graph(id="pressure-graph", style={"height": "30vh"}),
                    dcc.Graph(id="radius-graph", style={"height": "30vh"}),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Simulation time"),
                            dcc.Slider(
                                id="time-slider",
                                min=0,
                                max=NUM_SAMPLES - 1,
                                step=1,
                                value=0,
                            ),
                            html.Div(id="time-readout", style={"marginTop": "0.5rem"}),
                            html.Button("Animate bubble", id="animate-btn", n_clicks=0, style={"marginTop": "0.5rem"}),
                        ],
                        style={"flex": "2", "padding": "1rem", "border": "1px solid #ddd"},
                    ),
                    html.Div(
                        [dcc.Graph(id="bubble-figure", style={"height": "250px"})],
                        style={"flex": "1", "padding": "1rem", "border": "1px solid #ddd"},
                    ),
                ],
                style={"display": "flex", "gap": "1rem", "marginTop": "1rem"},
            ),
            dcc.Store(id="bubble-data"),
            dcc.Store(id="animate-state", data={"running": False, "index": 0}),
            dcc.Interval(id="animation-interval", interval=40, disabled=True),
        ]
    )


app = Dash(__name__)
app.layout = serve_layout


@app.callback(
    Output("pressure-graph", "figure"),
    Output("radius-graph", "figure"),
    Output("bubble-data", "data"),
    Output("time-slider", "max"),
    Output("time-slider", "value"),
    Input("shape-dropdown", "value"),
    Input("freq-slider", "value"),
    Input("pressure-slider", "value"),
    Input("radius-slider", "value"),
    Input("hann-check", "value"),
)
def update_simulation(shape, freq, pressure, radius, hann_values):
    result, ts_us, radius_um, pressure_kpa, metrics = simulate(
        shape,
        freq,
        pressure,
        radius,
        use_hann="hann" in (hann_values or []),
    )
    pressure_fig = build_line_figure(ts_us, pressure_kpa, "Time (μs)", "Pressure (kPa)", "Driving Pressure")
    radius_fig = build_line_figure(ts_us, radius_um, "Time (μs)", "Radius (μm)", "Radius")
    return (
        pressure_fig,
        radius_fig,
        {"ts": ts_us.tolist(), "radius": radius_um.tolist(), "metrics": metrics},
        len(ts_us) - 1,
        0,
    )


@app.callback(
    Output("bubble-figure", "figure"),
    Output("time-readout", "children"),
    Input("time-slider", "value"),
    Input("bubble-data", "data"),
)
def update_bubble_frame(frame_idx, data):
    if not data:
        return bubble_snapshot(4.0), ""
    idx = min(max(int(frame_idx or 0), 0), len(data["ts"]) - 1)
    radius = data["radius"][idx]
    ts = data["ts"][idx]
    return bubble_snapshot(radius), f"t = {ts:.2f} μs, R = {radius:.2f} μm"


@app.callback(
    Output("animate-state", "data"),
    Output("animation-interval", "disabled"),
    Input("animate-btn", "n_clicks"),
    State("animate-state", "data"),
)
def toggle_animation(n_clicks, state):
    if n_clicks is None:
        return state, True
    running = not state.get("running", False)
    return {"running": running, "index": 0}, not running


@app.callback(
    Output("time-slider", "value"),
    Output("animate-state", "data"),
    Input("animation-interval", "n_intervals"),
    State("animate-state", "data"),
    State("time-slider", "max"),
)
def advance_frame(_, state, max_idx):
    if not state.get("running"):
        return dash.no_update, state
    idx = (state.get("index", 0) + 1) % (int(max_idx) + 1)
    return idx, {"running": True, "index": idx}


if __name__ == "__main__":
    app.run(debug=True)
