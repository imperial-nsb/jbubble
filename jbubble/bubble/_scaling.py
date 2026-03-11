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

_FIELD_SCALES: dict[str, str] = {
    # Lengths
    "R0": "L_scale",
    "h": "L_scale",
    "d_s": "L_scale",
    "R_buckle": "L_scale",
    "tube_radius": "L_scale",
    "tube_length": "L_scale",
    "vessel_radius": "L_scale",
    "vessel_d": "L_scale",
    "tissue_d": "L_scale",
    # Pressures / elastic moduli
    "P_amb": "P_scale",
    "P_gas0": "P_scale",
    "G": "P_scale",
    "G_s": "P_scale",
    "vessel_E": "P_scale",
    # Surface tensions / shell elasticity
    "chi": "chi_scale",
    "sigma_L": "sigma_scale",
    "sigma_water": "sigma_scale",
    "sigma_break": "sigma_scale",
    # Dynamic viscosities
    "mu": "mu_scale",
    "mu_L": "mu_scale",
    "mu_s": "mu_scale",
    # Shell surface-dilatational viscosity
    "kappa_s": "kappa_scale",
    # Densities
    "rho_L": "rho_scale",
    "vessel_rho": "rho_scale",
    "tissue_rho": "rho_scale",
    # Velocities
    "c_L": "vel_scale",
    # Dimensionless
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
