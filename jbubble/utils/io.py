"""Save and load (batched) SimulationResults via orbax."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..bubble import (
    Bubble,
    ChurchGompertz,
    KellerMiksisGompertz,
    KelvinVoigtGompertz,
    LeightonGompertz,
    Marmottant,
    MarmottantGompertz,
    NeoHookeanGompertz,
    RayleighPlesset,
    SphericalConfinement,
)
from ..pulse import Pulse
from ..shapes import (
    Asymmetrical,
    InvertedSawtooth,
    Mono95,
    Mono99,
    NegativeQuadratic,
    Pulse9,
    Pulse10,
    PulseShape,
    Quadratic,
    Rect25,
    Rect25NegPos,
    Rect75,
    Rect75NegPos,
    Rect95,
    Sawtooth,
    Sine,
    SlantedSine,
    Square,
    SquareNegPos,
    TimeDomainSawtooth,
    TimeDomainSquare,
    TimeDomainTriangle,
    Triangle,
)
from ..simulation import SimulationResult
from ..units import Units

_BUBBLE_REGISTRY: dict[str, type[Bubble]] = {
    cls.__name__: cls
    for cls in [
        RayleighPlesset,
        Marmottant,
        MarmottantGompertz,
        KelvinVoigtGompertz,
        NeoHookeanGompertz,
        KellerMiksisGompertz,
        ChurchGompertz,
        LeightonGompertz,
        SphericalConfinement,
    ]
}

_SHAPE_REGISTRY: dict[str, type[PulseShape]] = {
    cls.__name__: cls
    for cls in [
        Sine,
        Sawtooth,
        InvertedSawtooth,
        Triangle,
        Quadratic,
        NegativeQuadratic,
        Asymmetrical,
        SlantedSine,
        Square,
        TimeDomainSquare,
        TimeDomainSawtooth,
        TimeDomainTriangle,
        Pulse9,
        Pulse10,
        Rect75,
        Rect25,
        Mono99,
        Mono95,
        Rect75NegPos,
        Rect25NegPos,
        SquareNegPos,
        Rect95,
    ]
}

# Fields on SimulationResult that are always present.
_ARRAY_FIELDS = (
    "ts",
    "radius",
    "radial_velocity",
    "radial_acceleration",
    "driving_pressure",
    "converged",
)
# Fields that are only present for vessel models.
_VESSEL_FIELDS = ("vessel_radius", "vessel_velocity")


def save(
    path: str | Path,
    result: SimulationResult,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a (batched) :class:`SimulationResult` to *path* using orbax.

    Parameters
    ----------
    path : str or Path
        Directory to create.  Must not already exist (orbax requirement).
    result : SimulationResult
        A single or batched (vmapped) simulation result.
    metadata : dict, optional
        Arbitrary JSON-serialisable metadata stored alongside the arrays.
    """
    import orbax.checkpoint as ocp

    path = Path(path).resolve()

    # -- Build orbax state dict ------------------------------------------------
    state: dict[str, Any] = {}
    for name in _ARRAY_FIELDS:
        state[name] = getattr(result, name)

    has_vessel = result.vessel_radius is not None
    if has_vessel:
        for name in _VESSEL_FIELDS:
            state[name] = getattr(result, name)

    # Bubble fields (all scalars or vmapped arrays).
    state["bubble"] = {
        f.name: getattr(result.bubble, f.name)
        for f in dataclasses.fields(result.bubble)
    }

    # Pulse fields — skip `shape` (nested module) and `apply_hann` (static bool).
    state["pulse"] = {
        f.name: getattr(result.pulse, f.name)
        for f in dataclasses.fields(result.pulse)
        if f.name not in ("shape", "apply_hann")
    }

    # PulseShape fields (usually empty; TimeDomainSquare has `sharpness`).
    shape_fields = {
        f.name: getattr(result.pulse.shape, f.name)
        for f in dataclasses.fields(result.pulse.shape)
    }
    if shape_fields:
        state["shape"] = shape_fields

    # Units (always three scalars).
    state["units"] = {
        f.name: getattr(result.units, f.name) for f in dataclasses.fields(result.units)
    }

    # -- Write orbax checkpoint ------------------------------------------------
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path / "state", state)

    # -- Write sidecar JSON ----------------------------------------------------
    meta = {
        "bubble_class": type(result.bubble).__name__,
        "pulse_shape_class": type(result.pulse.shape).__name__,
        "apply_hann": bool(result.pulse.apply_hann),
        "has_vessel": has_vessel,
        "metadata": metadata or {},
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def load(path: str | Path) -> tuple[SimulationResult, dict[str, Any]]:
    """Load a :class:`SimulationResult` saved by :func:`save`.

    Parameters
    ----------
    path : str or Path
        Checkpoint directory previously written by :func:`save`.

    Returns
    -------
    result : SimulationResult
        The restored simulation result (arrays are JAX arrays).
    metadata : dict
        The user metadata that was saved alongside.
    """
    import orbax.checkpoint as ocp

    path = Path(path).resolve()

    # -- Read sidecar ----------------------------------------------------------
    meta = json.loads((path / "meta.json").read_text())
    bubble_cls = _BUBBLE_REGISTRY[meta["bubble_class"]]
    shape_cls = _SHAPE_REGISTRY[meta["pulse_shape_class"]]

    # -- Restore orbax state ---------------------------------------------------
    with ocp.StandardCheckpointer() as ckptr:
        state = ckptr.restore(path / "state")

    # -- Helpers to convert numpy scalars/arrays back to jax/python types ------
    def _to_jax(v: Any) -> Any:
        """Convert numpy arrays to jax arrays, leave Python scalars alone."""
        if isinstance(v, np.ndarray):
            return jnp.asarray(v)
        return v

    def _to_python(v: Any) -> Any:
        """Convert numpy scalars to Python scalars for module constructors."""
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return v.item()
        if isinstance(v, np.ndarray):
            return jnp.asarray(v)
        return v

    # -- Reconstruct modules ---------------------------------------------------
    shape_dict = {k: _to_python(v) for k, v in state.get("shape", {}).items()}
    pulse_shape = shape_cls(**shape_dict)

    pulse_dict = {k: _to_python(v) for k, v in state["pulse"].items()}
    pulse = Pulse(shape=pulse_shape, apply_hann=meta["apply_hann"], **pulse_dict)

    bubble_dict = {k: _to_python(v) for k, v in state["bubble"].items()}
    bubble = bubble_cls(**bubble_dict)

    units_dict = {k: _to_python(v) for k, v in state["units"].items()}
    units = Units(**units_dict)

    # -- Assemble SimulationResult ---------------------------------------------
    result = SimulationResult(
        ts=_to_jax(state["ts"]),
        radius=_to_jax(state["radius"]),
        radial_velocity=_to_jax(state["radial_velocity"]),
        radial_acceleration=_to_jax(state["radial_acceleration"]),
        driving_pressure=_to_jax(state["driving_pressure"]),
        converged=_to_jax(state["converged"]),
        vessel_radius=_to_jax(state["vessel_radius"]) if meta["has_vessel"] else None,
        vessel_velocity=(
            _to_jax(state["vessel_velocity"]) if meta["has_vessel"] else None
        ),
        bubble=bubble,
        pulse=pulse,
        units=units,
    )

    return result, meta["metadata"]
