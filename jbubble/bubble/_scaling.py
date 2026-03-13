"""Non-dimensionalisation utilities for Equinox modules.

``_FIELD_SCALES`` maps dataclass field names to ``Units`` attribute names.
``_scale_module`` recursively non-dimensionalises any Equinox module by
dividing each field by its appropriate scale factor.
"""

import dataclasses
from typing import Any

import equinox as eqx

# ---------------------------------------------------------------------------
# Field-scaling registry
# ---------------------------------------------------------------------------
# Maps field names to their corresponding ``Units`` attribute.
#
# Notes:
#  - Fields whose types are Property / eqx.Module subclasses are handled
#    by the recursive branch in _scale_module and do NOT need an entry here.
#  - Dimensionless fields (gamma, nu, h_frac, ...) are not listed; they fall
#    through to the identity branch.

_FIELD_SCALES: dict[str, str] = {
    # Lengths
    "R0": "L_scale",
    "d_s": "L_scale",
    "R_buckle": "L_scale",
    "tube_radius": "L_scale",
    "tube_length": "L_scale",
    "vessel_radius": "L_scale",
    "vessel_d": "L_scale",
    "tissue_d": "L_scale",
    # Pressures / elastic moduli
    "P_amb": "P_scale",
    "vessel_E": "P_scale",
    # Surface tensions / shell elasticity
    "chi": "chi_scale",
    "sigma_break": "sigma_scale",
    "sigma_rupture": "sigma_scale",
    # Densities
    "rho_L": "rho_scale",
    "vessel_rho": "rho_scale",
    "tissue_rho": "rho_scale",
    # Velocities
    "c_L": "vel_scale",
    # Dimensionless (explicit no-op entries for documentation)
    "gamma": "unit_scale",
}


def _scale_module(module: eqx.Module, units: Any) -> eqx.Module:
    """Non-dimensionalise an Equinox module by scaling all dataclass fields.

    Sub-modules (GasModel, ShellModel, etc.) are recursively scaled.
    Scalar fields are divided by the appropriate unit scale from
    ``_FIELD_SCALES``.  Fields not in the registry are left unchanged.

    ``ConstantProperty`` instances are handled specially: their ``val``
    field is scaled using the ``_scale`` attribute which names the
    appropriate ``Units`` property.
    """
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(module):
        val = getattr(module, f.name)
        if isinstance(val, eqx.Module):
            kwargs[f.name] = _scale_module(val, units)
        elif f.name in _FIELD_SCALES:
            kwargs[f.name] = val / getattr(units, _FIELD_SCALES[f.name])
        elif f.name == "val" and hasattr(module, "_scale"):
            kwargs[f.name] = val / getattr(units, module._scale)
        else:
            kwargs[f.name] = val
    return type(module)(**kwargs)
