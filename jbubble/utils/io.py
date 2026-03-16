"""Save and load (batched) SimulationResults via orbax."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..bubble.eom import (
    KellerMiksis,
    LeightonTube,
    ModifiedRayleighPlesset,
    RayleighPlesset,
    SphericalConfinement,
)
from ..bubble.gas import PolytropicGas, VanDerWaalsGas
from ..bubble.medium import (
    KelvinVoigtMedium,
    NeoHookeanMedium,
    NewtonianMedium,
)
from ..bubble.property import ConstantProperty
from ..bubble.shell import (
    GompertzSurfaceTension,
    LipidShell,
    MarmottantSurfaceTension,
    NoShell,
    ThickShell,
)
from ..bubble.state import BubbleState, ConfinedBubbleState
from ..pulse.chirp import ChirpPulse
from ..pulse.envelope import (
    HannEnvelope,
    RectangularEnvelope,
    TukeyEnvelope,
)
from ..pulse.sampled import SampledPulse
from ..pulse.shapes import (
    Asymmetrical,
    InvertedSawtooth,
    Mono95,
    Mono99,
    NegativeQuadratic,
    Pulse9,
    Pulse10,
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
from ..pulse.tone_burst import ToneBurst
from ..simulation import SimulationResult

# Registry of all concrete classes that can be serialised.
_MODULE_REGISTRY: dict[str, type[eqx.Module]] = {
    cls.__name__: cls
    for cls in [
        # States
        BubbleState,
        ConfinedBubbleState,
        # EoMs
        RayleighPlesset,
        ModifiedRayleighPlesset,
        KellerMiksis,
        LeightonTube,
        SphericalConfinement,
        # Gas
        PolytropicGas,
        VanDerWaalsGas,
        # Properties
        ConstantProperty,
        GompertzSurfaceTension,
        MarmottantSurfaceTension,
        # Shell
        NoShell,
        LipidShell,
        ThickShell,
        # Medium
        NewtonianMedium,
        KelvinVoigtMedium,
        NeoHookeanMedium,
        # Envelopes
        RectangularEnvelope,
        HannEnvelope,
        TukeyEnvelope,
        # Pulse types
        ToneBurst,
        SampledPulse,
        ChirpPulse,
        # Shapes
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

# Backward-compat: files saved before the Property→ConstantProperty rename.
_MODULE_REGISTRY["Property"] = ConstantProperty


def _serialise_module(module: eqx.Module) -> dict[str, Any]:
    """Recursively serialise an Equinox module to a nested dict."""
    result: dict[str, Any] = {}
    for f in dataclasses.fields(module):
        val = getattr(module, f.name)
        if isinstance(val, eqx.Module):
            result[f.name] = _serialise_module(val)
        else:
            result[f.name] = val
    return result


def _collect_class_names(module: eqx.Module) -> dict[str, Any]:
    """Recursively collect the class name of each sub-module."""
    result: dict[str, str | dict] = {"__class__": type(module).__name__}
    for f in dataclasses.fields(module):
        val = getattr(module, f.name)
        if isinstance(val, eqx.Module):
            result[f.name] = _collect_class_names(val)
    return result


def _deserialise_module(
    data: dict[str, Any],
    class_info: dict[str, Any],
) -> eqx.Module:
    """Recursively reconstruct an Equinox module from serialised data."""
    cls = _MODULE_REGISTRY[class_info["__class__"]]
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        val = data[f.name]
        if f.name in class_info and isinstance(class_info[f.name], dict):
            kwargs[f.name] = _deserialise_module(val, class_info[f.name])
        else:
            kwargs[f.name] = _to_python(val)
    return cls(**kwargs)


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


def save(
    path: str | Path,
    result: SimulationResult,
    *,
    eom: eqx.Module | None = None,
    pulse: eqx.Module | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a (batched) :class:`SimulationResult` to *path* using orbax.

    Parameters
    ----------
    path : str or Path
        Directory to create.  Must not already exist (orbax requirement).
    result : SimulationResult
        A single or batched (vmapped) simulation result.
    eom : EquationOfMotion, optional
        Equation of motion to save as provenance.  Not required for
        array-only workflows (e.g. parameter sweeps with varying EoMs).
    pulse : Pulse, optional
        Driving pulse to save as provenance.
    metadata : dict, optional
        Arbitrary JSON-serialisable metadata stored alongside the arrays.
    """
    import orbax.checkpoint as ocp

    path = Path(path).resolve()

    state_cls_name = type(result.state).__name__

    # -- Build orbax checkpoint dict ------------------------------------------
    ckpt: dict[str, Any] = {
        "ts": result.ts,
        "driving_pressure": result.driving_pressure,
        "converged": result.converged,
        "state": _serialise_module(result.state),
        "state_dot": _serialise_module(result.state_dot),
    }
    if eom is not None:
        ckpt["eom"] = _serialise_module(eom)
    if pulse is not None:
        ckpt["pulse"] = _serialise_module(pulse)

    # -- Write orbax checkpoint ------------------------------------------------
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path / "state", ckpt)

    # -- Write sidecar JSON ----------------------------------------------------
    meta: dict[str, Any] = {
        "state_class": state_cls_name,
        "metadata": metadata or {},
    }
    if eom is not None:
        meta["eom_classes"] = _collect_class_names(eom)
    if pulse is not None:
        meta["pulse_classes"] = _collect_class_names(pulse)
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
    info : dict
        Provenance and user metadata with keys:

        * ``"metadata"`` — the user-supplied dict passed to :func:`save`.
        * ``"eom"`` — reconstructed EoM, or ``None`` if not saved.
        * ``"pulse"`` — reconstructed Pulse, or ``None`` if not saved.
    """
    import orbax.checkpoint as ocp

    path = Path(path).resolve()

    # -- Read sidecar ----------------------------------------------------------
    meta = json.loads((path / "meta.json").read_text())

    # -- Restore orbax checkpoint ----------------------------------------------
    with ocp.StandardCheckpointer() as ckptr:
        ckpt = ckptr.restore(path / "state")

    # -- Reconstruct state PyTrees ---------------------------------------------
    loaded_state = _deserialise_module(
        ckpt["state"], {"__class__": meta["state_class"]}
    )
    loaded_state_dot = _deserialise_module(
        ckpt["state_dot"], {"__class__": meta["state_class"]}
    )

    # -- Reconstruct optional provenance ---------------------------------------
    loaded_eom = None
    if "eom_classes" in meta and "eom" in ckpt:
        loaded_eom = _deserialise_module(ckpt["eom"], meta["eom_classes"])

    loaded_pulse = None
    if "pulse_classes" in meta and "pulse" in ckpt:
        loaded_pulse = _deserialise_module(ckpt["pulse"], meta["pulse_classes"])

    # -- Assemble SimulationResult ---------------------------------------------
    result = SimulationResult(
        ts=_to_jax(ckpt["ts"]),
        state=loaded_state,
        state_dot=loaded_state_dot,
        driving_pressure=_to_jax(ckpt["driving_pressure"]),
        converged=_to_jax(ckpt["converged"]),
    )

    info = {
        "eom": loaded_eom,
        "pulse": loaded_pulse,
        "metadata": meta["metadata"],
    }
    return result, info
