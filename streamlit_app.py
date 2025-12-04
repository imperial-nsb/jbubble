"""Simple Streamlit interface for jbubble simulations."""

from __future__ import annotations

import jax
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from jbubble import (
    Units,
    SaveSpec,
    arrays_from_result,
    build_pulse,
    default_bubble,
    line_trace,
    run_simulation,
)

UNITS = Units()
SAVE_SPEC = SaveSpec(num_samples=1000)
MIN_FREQ_KHZ = 250.0
MAX_FREQ_KHZ = 1500.0
PRESSURE_LIMIT_KPA = 500.0
RADIUS_AXIS_MAX_UM = 15.0
TIME_MAX_US = 15.0
DEFAULTS = {
    "freq": (MAX_FREQ_KHZ + MIN_FREQ_KHZ) / 2,
    "pressure": PRESSURE_LIMIT_KPA / 2,
    "radius": 3.0,
    "cycles": 3,
}

@st.cache_resource(show_spinner=False)
def _get_simulator():
    return jax.jit(run_simulation)


JIT_SIM = _get_simulator()


def simulate(freq_khz: float, pressure_kpa: float, radius_um: float, cycles: int):
    bubble = default_bubble(R0=radius_um * 1e-6)
    pulse = build_pulse(
        "sine",
        freq=freq_khz * 1e3,
        pressure=pressure_kpa * 1e3,
        cycle_num=cycles,
        initial_time=1e-6,
        apply_hann=False,
    )
    result = JIT_SIM(
        bubble=bubble,
        pulse=pulse,
        units=UNITS,
        save_spec=SAVE_SPEC,
    )
    return result, arrays_from_result(result)


def _stacked_figure(arrays, marker_idx: int | None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(line_trace(arrays.time_us, arrays.pressure_kpa, name="Driving Pressure", color="#45FFE9"), row=1, col=1)
    fig.add_trace(line_trace(arrays.time_us, arrays.radius_um, name="Bubble Radius", color="#FFCC33"), row=2, col=1)
    fig.update_xaxes(range=(0.0, TIME_MAX_US), title_text="Time (μs)", row=2, col=1)
    fig.update_xaxes(range=(0.0, TIME_MAX_US), row=1, col=1, showticklabels=False)
    fig.update_yaxes(range=(-PRESSURE_LIMIT_KPA, PRESSURE_LIMIT_KPA), title_text="Pressure (kPa)", row=1, col=1)
    fig.update_yaxes(range=(0.0, RADIUS_AXIS_MAX_UM), title_text="Radius (μm)", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=20, t=30, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


st.set_page_config(page_title="jbubble", layout="wide")

with st.sidebar:
    st.image("jbubble.svg", use_container_width=True)
    st.write("")
    freq = st.slider("Frequency (kHz)", min_value=MIN_FREQ_KHZ, max_value=MAX_FREQ_KHZ, value=DEFAULTS["freq"], step=10.0)
    st.write("")
    pressure = st.slider("Pressure amplitude (kPa)", min_value=0.0, max_value=PRESSURE_LIMIT_KPA, value=DEFAULTS["pressure"], step=10.0)
    st.write("")
    radius = st.slider("Equilibrium radius (μm)", min_value=1.0, max_value=5.0, value=DEFAULTS["radius"], step=0.1)
    st.write("")
    cycles = st.slider("Pulse cycles", min_value=2, max_value=10, value=DEFAULTS["cycles"], step=1)
    st.write("")

result, arrays = simulate(freq, pressure, radius, cycles)

with st.sidebar:
    converged = bool(result.converged)
    st.write("")
    with st.status("Solver status", state="complete" if converged else "error"):
        st.write("✅ Converged" if converged else "🔴 Max steps!")
    st.caption("\nBubble dynamics simulation powered by JAX")

st.plotly_chart(_stacked_figure(arrays, None), use_container_width=True)
col1, col2, col3, col4 = st.columns(4)
st.write("\n")
col1.metric("Max Radius (μm)", f"{arrays.radius_um.max():.2f}")
col2.metric("Min Radius (μm)", f"{arrays.radius_um.min():.2f}")
col3.metric("Max Expansion Ratio", f"{(arrays.radius_um.max() / (radius)):.2f}")
col4.metric("Collapse Ratio", f"{(arrays.radius_um.min() / (radius)):.2f}")
