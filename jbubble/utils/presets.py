"""Preset bubble configurations for common simulation scenarios.

Provides factory functions that assemble ready-to-run ``(eom, pulse)`` pairs
for the most common microbubble simulation use cases.  Each function exposes
the parameters that vary most in practice and uses physically reasonable
defaults for everything else.

Quick-start example::

    from jbubble import run_simulation
    from jbubble.utils.presets import free_bubble
    import jax

    eom, pulse = free_bubble(R0=3e-6, pressure=200e3)
    result = jax.jit(run_simulation)(eom, pulse)

Available presets
-----------------
free_bubble
    Uncoated polytropic gas bubble in a Newtonian liquid.
    Simplest case — good starting point for sensitivity studies.

lipid_bubble
    Thin lipid monolayer shell with smooth Gompertz surface tension.
    Representative of clinical UCAs such as SonoVue or Definity.

thick_shell_bubble
    Thick viscoelastic polymer shell (Church 1995 model).
    Representative of polymer-shelled agents such as Optison.
"""

from __future__ import annotations

from typing import NamedTuple

from ..bubble.eom import EquationOfMotion, KellerMiksis
from ..bubble.gas import PolytropicGas
from ..bubble.medium import NewtonianMedium
from ..bubble.shell import GompertzSurfaceTension, LipidShell, NoShell, ThickShell
from ..pulse.base import Pulse
from ..pulse.shapes import Sine
from ..pulse.tone_burst import ToneBurst


class BubblePreset(NamedTuple):
    """Assembled (EoM, pulse) pair returned by preset factory functions.

    Can be unpacked directly::

        eom, pulse = free_bubble()

    or accessed by name::

        preset = free_bubble()
        result = run_simulation(preset.eom, preset.pulse)
    """

    eom: EquationOfMotion
    pulse: Pulse


def free_bubble(
    R0: float = 2e-6,
    freq: float = 1e6,
    pressure: float = 100e3,
    cycle_num: int = 5,
    *,
    gamma: float = 1.4,
    sigma: float = 0.072,
    mu: float = 1e-3,
    P_amb: float = 101325.0,
    rho_L: float = 998.0,
    c_L: float = 1500.0,
) -> BubblePreset:
    """Uncoated gas bubble in a Newtonian liquid (Keller-Miksis).

    The simplest physically meaningful preset: a polytropic gas bubble
    with Laplace surface tension only — no shell coating.  Suitable for
    free cavitation bubbles and as a baseline before adding a shell.

    Physics: Keller-Miksis EoM + PolytropicGas + NoShell + NewtonianMedium.

    Parameters
    ----------
    R0 : float
        Equilibrium radius [m].  Default: 2 µm.
    freq : float
        Driving frequency [Hz].  Default: 1 MHz.
    pressure : float
        Peak acoustic pressure amplitude [Pa].  Default: 100 kPa.
    cycle_num : int
        Number of tone-burst cycles.  Default: 5.
    gamma : float
        Polytropic exponent.  Default: 1.4 (diatomic gas / air).
    sigma : float
        Surface tension [N/m].  Default: 0.072 (air–water interface).
    mu : float
        Liquid dynamic viscosity [Pa·s].  Default: 1e-3 (water, 20 °C).
    P_amb : float
        Ambient pressure [Pa].  Default: 101 325 (1 atm).
    rho_L : float
        Liquid density [kg/m³].  Default: 998 (water, 20 °C).
    c_L : float
        Speed of sound in the liquid [m/s].  Default: 1500 (water).

    Returns
    -------
    BubblePreset
        ``(eom, pulse)`` ready for :func:`~jbubble.run_simulation`.
    """
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=gamma),
        shell=NoShell(sigma=sigma),
        medium=NewtonianMedium(mu=mu),
        R0=R0,
        P_amb=P_amb,
        rho_L=rho_L,
        c_L=c_L,
    )
    pulse = ToneBurst(freq=freq, pressure=pressure, shape=Sine(), cycle_num=cycle_num)
    return BubblePreset(eom=eom, pulse=pulse)


def lipid_bubble(
    R0: float = 2e-6,
    freq: float = 1e6,
    pressure: float = 100e3,
    cycle_num: int = 5,
    *,
    kappa_s: float = 2.4e-9,
    chi: float = 0.55,
    sigma_rupture: float = 0.072,
    R_buckle_ratio: float = 0.98,
    gamma: float = 1.4,
    mu: float = 1e-3,
    P_amb: float = 101325.0,
    rho_L: float = 998.0,
    c_L: float = 1500.0,
) -> BubblePreset:
    """Lipid-shelled ultrasound contrast agent — Marmottant (2005) model.

    Models a thin lipid monolayer with surface-dilatational viscosity and a
    smooth Gompertz surface tension law (preferred over the piecewise
    Marmottant law for gradient-based parameter fitting).  Representative
    of clinical agents such as SonoVue and Definity.

    Physics: Keller-Miksis EoM + PolytropicGas
             + LipidShell(GompertzSurfaceTension) + NewtonianMedium.

    Parameters
    ----------
    R0 : float
        Equilibrium radius [m].  Default: 2 µm.
    freq : float
        Driving frequency [Hz].  Default: 1 MHz.
    pressure : float
        Peak acoustic pressure amplitude [Pa].  Default: 100 kPa.
    cycle_num : int
        Number of tone-burst cycles.  Default: 5.
    kappa_s : float
        Shell surface-dilatational viscosity [N·s/m].  Default: 2.4e-9
        (Marmottant 2005, BR14 / SonoVue-type).
    chi : float
        Shell elasticity [N/m].  Default: 0.55.
    sigma_rupture : float
        Asymptotic (ruptured) surface tension [N/m].  Default: 0.072 (water).
    R_buckle_ratio : float
        Buckling radius as fraction of R0.  Default: 0.98 — gives an
        initial surface tension ``sigma(R0) ≈ 0.023 N/m``, well below
        ``sigma_rupture``, which is required for the Gompertz model to
        be well-posed.
    gamma : float
        Polytropic exponent.  Default: 1.4.
    mu : float
        Liquid dynamic viscosity [Pa·s].  Default: 1e-3 (water).
    P_amb : float
        Ambient pressure [Pa].  Default: 101 325 (1 atm).
    rho_L : float
        Liquid density [kg/m³].  Default: 998 (water).
    c_L : float
        Speed of sound in the liquid [m/s].  Default: 1500 (water).

    Returns
    -------
    BubblePreset
        ``(eom, pulse)`` ready for :func:`~jbubble.run_simulation`.

    References
    ----------
    Marmottant et al., J. Acoust. Soc. Am. 118 (2005) 3499–3505.
    """
    shell = LipidShell(
        sigma=GompertzSurfaceTension(
            R_buckle_ratio=R_buckle_ratio,
            chi=chi,
            sigma_rupture=sigma_rupture,
        ),
        kappa_s=kappa_s,
    )
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=gamma),
        shell=shell,
        medium=NewtonianMedium(mu=mu),
        R0=R0,
        P_amb=P_amb,
        rho_L=rho_L,
        c_L=c_L,
    )
    pulse = ToneBurst(freq=freq, pressure=pressure, shape=Sine(), cycle_num=cycle_num)
    return BubblePreset(eom=eom, pulse=pulse)


def thick_shell_bubble(
    R0: float = 2e-6,
    freq: float = 1e6,
    pressure: float = 100e3,
    cycle_num: int = 5,
    *,
    d_s: float = 15e-9,
    G_s: float = 10e6,
    mu_s: float = 0.5,
    sigma: float = 0.04,
    gamma: float = 1.4,
    mu: float = 1e-3,
    P_amb: float = 101325.0,
    rho_L: float = 998.0,
    c_L: float = 1500.0,
) -> BubblePreset:
    """Polymer thick-shell bubble — Church (1995) model.

    Models a thick viscoelastic shell with elastic and viscous contributions.
    Representative of polymer-shelled agents such as Optison (albumin),
    or experimental PLGA microbubbles.

    Physics: Keller-Miksis EoM + PolytropicGas + ThickShell + NewtonianMedium.

    Parameters
    ----------
    R0 : float
        Equilibrium radius [m].  Default: 2 µm.
    freq : float
        Driving frequency [Hz].  Default: 1 MHz.
    pressure : float
        Peak acoustic pressure amplitude [Pa].  Default: 100 kPa.
    cycle_num : int
        Number of tone-burst cycles.  Default: 5.
    d_s : float
        Shell thickness [m].  Default: 15 nm.
    G_s : float
        Shell shear modulus [Pa].  Default: 10 MPa (stiff polymer shell).
    mu_s : float
        Shell viscosity [Pa·s].  Default: 0.5.
    sigma : float
        Surface tension [N/m].  Default: 0.04 (reduced by polymer coating).
    gamma : float
        Polytropic exponent.  Default: 1.4.
    mu : float
        Liquid dynamic viscosity [Pa·s].  Default: 1e-3 (water).
    P_amb : float
        Ambient pressure [Pa].  Default: 101 325 (1 atm).
    rho_L : float
        Liquid density [kg/m³].  Default: 998 (water).
    c_L : float
        Speed of sound in the liquid [m/s].  Default: 1500 (water).

    Returns
    -------
    BubblePreset
        ``(eom, pulse)`` ready for :func:`~jbubble.run_simulation`.

    References
    ----------
    Church, J. Acoust. Soc. Am. 97 (1995) 1510–1521.
    """
    eom = KellerMiksis(
        gas=PolytropicGas(gamma=gamma),
        shell=ThickShell(sigma=sigma, d_s=d_s, G_s=G_s, mu_s=mu_s),
        medium=NewtonianMedium(mu=mu),
        R0=R0,
        P_amb=P_amb,
        rho_L=rho_L,
        c_L=c_L,
    )
    pulse = ToneBurst(freq=freq, pressure=pressure, shape=Sine(), cycle_num=cycle_num)
    return BubblePreset(eom=eom, pulse=pulse)
