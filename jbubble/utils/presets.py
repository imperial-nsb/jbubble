"""Preset bubble configurations for common simulation scenarios.

The factory functions in this module assemble ready-to-run `(eom, pulse)`
pairs for the most common microbubble simulation use cases. Each function
exposes the parameters that vary most in practice and uses physically
reasonable defaults for everything else.

The available presets are:

- [`free_bubble`][jbubble.utils.presets.free_bubble]: an uncoated
  polytropic gas bubble in a Newtonian liquid. The simplest case, and a
  good starting point for sensitivity studies.
- [`lipid_bubble`][jbubble.utils.presets.lipid_bubble]: a thin lipid
  monolayer shell with the smoothed Marmottant surface tension law and the
  SonoVue parameters of Gümmer et al. (2021). Representative of clinical
  ultrasound contrast agents (UCAs) such as SonoVue.
- [`thick_shell_bubble`][jbubble.utils.presets.thick_shell_bubble]: a
  thick viscoelastic polymer shell (Church 1995 model). Representative of
  polymer-shelled agents such as Optison.

All three presets share the same liquid: water at about 20 °C, with
properties from the NIST Chemistry WebBook.

Examples
--------
```python
import jax

from jbubble import run_simulation
from jbubble.utils.presets import free_bubble

eom, pulse = free_bubble(R0=3e-6, pressure=200e3)
result = jax.jit(run_simulation)(eom, pulse)
```
"""

from __future__ import annotations

from typing import NamedTuple

from ..bubble.eom import EquationOfMotion, KellerMiksis
from ..bubble.gas import PolytropicGas
from ..bubble.medium import NewtonianMedium
from ..bubble.shell import (
    LipidShell,
    NoShell,
    SmoothMarmottantSurfaceTension,
    ThickShell,
)
from ..pulse.base import Pulse
from ..pulse.shapes import Sine
from ..pulse.tone_burst import ToneBurst

__all__ = ["BubblePreset", "free_bubble", "lipid_bubble", "thick_shell_bubble"]


class BubblePreset(NamedTuple):
    """Assembled `(eom, pulse)` pair that the preset factory functions return.

    You can unpack it directly:

    ```python
    eom, pulse = free_bubble()
    ```

    Or you can access the fields by name:

    ```python
    preset = free_bubble()
    result = run_simulation(preset.eom, preset.pulse)
    ```

    Attributes
    ----------
    eom : EquationOfMotion
        Assembled equation of motion.
    pulse : Pulse
        Driving pulse.
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

    The simplest physically meaningful preset: a polytropic gas bubble with
    Laplace surface tension only and no shell coating. Suitable for free
    cavitation bubbles, and as a baseline before you add a shell.

    Physics: [`KellerMiksis`][jbubble.bubble.eom.KellerMiksis] +
    [`PolytropicGas`][jbubble.bubble.gas.PolytropicGas] +
    [`NoShell`][jbubble.bubble.shell.NoShell] +
    [`NewtonianMedium`][jbubble.bubble.medium.NewtonianMedium], driven by
    a sine [`ToneBurst`][jbubble.pulse.tone_burst.ToneBurst].

    Parameters
    ----------
    R0 : float
        Equilibrium radius [m]. Default: `2e-6` (2 µm).
    freq : float
        Driving frequency [Hz]. Default: `1e6` (1 MHz).
    pressure : float
        Peak acoustic pressure amplitude [Pa]. Default: `100e3` (100 kPa).
    cycle_num : int
        Number of tone-burst cycles. Default: `5`.
    gamma : float
        Polytropic exponent. Default: `1.4`, the adiabatic exponent of a
        diatomic gas such as air.
    sigma : float
        Surface tension [N/m]. Default: `0.072` (air-water interface at
        20 °C: 0.0727 N/m, NIST).
    mu : float
        Liquid dynamic viscosity [Pa s]. Default: `1e-3` (water at 20 °C:
        1.0016e-3 Pa s, NIST).
    P_amb : float
        Ambient pressure [Pa]. Default: `101325.0` (one standard
        atmosphere).
    rho_L : float
        Liquid density [kg/m³]. Default: `998.0` (water at 20 °C:
        998.2 kg/m³, NIST).
    c_L : float
        Speed of sound in the liquid [m/s]. Default: `1500.0`, the round
        value that ultrasound modelling commonly uses for water (NIST gives
        1482 m/s at 20 °C and 1497 m/s at 25 °C).

    Returns
    -------
    BubblePreset
        `(eom, pulse)` pair, ready for
        [`run_simulation`][jbubble.simulation.run_simulation].

    References
    ----------
    NIST Chemistry WebBook, NIST Standard Reference Database 69,
    Thermophysical Properties of Fluid Systems: water at 0.101325 MPa.
    [doi:10.18434/T4D303](https://doi.org/10.18434/T4D303)
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
    kappa_s: float = 7.5e-9,
    chi: float = 0.5,
    sigma_rupture: float = 0.072,
    R_buckle_ratio: float = 0.98058,
    smoothing: float = 0.01,
    gamma: float = 1.095,
    mu: float = 1e-3,
    P_amb: float = 101325.0,
    rho_L: float = 998.0,
    c_L: float = 1500.0,
) -> BubblePreset:
    r"""Lipid-shelled ultrasound contrast agent: the Marmottant (2005) model.

    Models an SF6 bubble coated with a thin lipid monolayer that buckles
    under compression, stretches elastically, and ruptures under
    expansion, with surface-dilatational viscosity. The surface tension
    follows
    [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension],
    which matches the piecewise Marmottant law to within
    `smoothing * ln(2) * sigma_rupture` and keeps gradients smooth for
    fitting.

    The shell and gas defaults are the SonoVue parameters of Gümmer et al.
    (2021), who take them from earlier characterisations of SonoVue. The
    preset solves the Keller-Miksis equation with a polytropic gas, not
    the Rayleigh-Plesset and Gilmore equations with a hard-core gas that
    Gümmer et al. use.

    Physics: [`KellerMiksis`][jbubble.bubble.eom.KellerMiksis] +
    [`PolytropicGas`][jbubble.bubble.gas.PolytropicGas] +
    [`LipidShell`][jbubble.bubble.shell.LipidShell] with
    [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension] +
    [`NewtonianMedium`][jbubble.bubble.medium.NewtonianMedium], driven by
    a sine [`ToneBurst`][jbubble.pulse.tone_burst.ToneBurst].

    Parameters
    ----------
    R0 : float
        Equilibrium radius [m]. Default: `2e-6` (2 µm, the top of the
        1-2 µm range that Gümmer et al. study).
    freq : float
        Driving frequency [Hz]. Default: `1e6` (1 MHz).
    pressure : float
        Peak acoustic pressure amplitude [Pa]. Default: `100e3` (100 kPa).
    cycle_num : int
        Number of tone-burst cycles. Default: `5`.
    kappa_s : float
        Shell surface-dilatational viscosity [N s/m]. Default: `7.5e-9`
        (Gümmer et al., who consider 5e-9 to 1e-8).
    chi : float
        Shell elasticity [N/m]. Default: `0.5` (Gümmer et al.).
    sigma_rupture : float
        Surface tension after rupture [N/m]. Default: `0.072`, the clean
        water surface (Gümmer et al.).
    R_buckle_ratio : float
        Buckling radius as a fraction of `R0`. Default: `0.98058`, which
        gives the surface tension at `R0` of Gümmer et al.,
        $\sigma_0 = 0.020$ N/m, through their Eq. 11:
        $(1 + \sigma_0/\chi)^{-1/2} = 0.98058$ at $\chi = 0.5$ N/m. If you
        change `chi`, $\sigma_0$ changes with it.
    smoothing : float
        Corner width of the smoothed Marmottant law as a fraction of
        `sigma_rupture`. Default: `0.01`; see
        [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension].
    gamma : float
        Polytropic exponent. Default: `1.095`, the value Gümmer et al. use
        for the SF6 in SonoVue.
    mu : float
        Liquid dynamic viscosity [Pa s]. Default: `1e-3` (water at 20 °C:
        1.0016e-3 Pa s, NIST).
    P_amb : float
        Ambient pressure [Pa]. Default: `101325.0` (one standard
        atmosphere).
    rho_L : float
        Liquid density [kg/m³]. Default: `998.0` (water at 20 °C:
        998.2 kg/m³, NIST).
    c_L : float
        Speed of sound in the liquid [m/s]. Default: `1500.0`, the round
        value that ultrasound modelling commonly uses for water (NIST gives
        1482 m/s at 20 °C and 1497 m/s at 25 °C).

    Returns
    -------
    BubblePreset
        `(eom, pulse)` pair, ready for
        [`run_simulation`][jbubble.simulation.run_simulation].

    References
    ----------
    Gümmer, J., Schenke, S., & Denner, F. (2021). Modelling lipid-coated
    microbubbles in focused ultrasound applications at subresonance
    frequencies. *Ultrasound Med. Biol.* 47(10), 2958-2979.
    [doi:10.1016/j.ultrasmedbio.2021.06.012](https://doi.org/10.1016/j.ultrasmedbio.2021.06.012)

    Marmottant, P., van der Meer, S., Emmer, M., Versluis, M., de Jong, N.,
    Hilgenfeldt, S., & Lohse, D. (2005). A model for large amplitude
    oscillations of coated bubbles accounting for buckling and rupture.
    *J. Acoust. Soc. Am.* 118(6), 3499-3505.
    [doi:10.1121/1.2109427](https://doi.org/10.1121/1.2109427)

    NIST Chemistry WebBook, NIST Standard Reference Database 69,
    Thermophysical Properties of Fluid Systems: water at 0.101325 MPa.
    [doi:10.18434/T4D303](https://doi.org/10.18434/T4D303)
    """
    shell = LipidShell(
        sigma=SmoothMarmottantSurfaceTension(
            R_buckle_ratio=R_buckle_ratio,
            chi=chi,
            sigma_rupture=sigma_rupture,
            smoothing=smoothing,
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
    """Polymer thick-shell bubble: the Church (1995) model.

    Models a thick viscoelastic shell with elastic and viscous
    contributions. Representative of polymer-shelled agents such as
    Optison (albumin), or experimental poly(lactic-co-glycolic acid) (PLGA)
    microbubbles.

    Physics: [`KellerMiksis`][jbubble.bubble.eom.KellerMiksis] +
    [`PolytropicGas`][jbubble.bubble.gas.PolytropicGas] +
    [`ThickShell`][jbubble.bubble.shell.ThickShell] +
    [`NewtonianMedium`][jbubble.bubble.medium.NewtonianMedium], driven by
    a sine [`ToneBurst`][jbubble.pulse.tone_burst.ToneBurst].

    Parameters
    ----------
    R0 : float
        Equilibrium radius [m]. Default: `2e-6` (2 µm).
    freq : float
        Driving frequency [Hz]. Default: `1e6` (1 MHz).
    pressure : float
        Peak acoustic pressure amplitude [Pa]. Default: `100e3` (100 kPa).
    cycle_num : int
        Number of tone-burst cycles. Default: `5`.
    d_s : float
        Shell thickness [m]. Default: `15e-9` (15 nm).
    G_s : float
        Shell shear modulus [Pa]. Default: `10e6` (10 MPa, a stiff polymer
        shell).
    mu_s : float
        Shell viscosity [Pa·s]. Default: `0.5`.
    sigma : float
        Surface tension [N/m]. Default: `0.04` (reduced by the polymer
        coating).
    gamma : float
        Polytropic exponent. Default: `1.4`.
    mu : float
        Liquid dynamic viscosity [Pa·s]. Default: `1e-3` (water).
    P_amb : float
        Ambient pressure [Pa]. Default: `101325.0` (1 atm).
    rho_L : float
        Liquid density [kg/m³]. Default: `998.0` (water).
    c_L : float
        Speed of sound in the liquid [m/s]. Default: `1500.0` (water).

    Returns
    -------
    BubblePreset
        `(eom, pulse)` pair, ready for
        [`run_simulation`][jbubble.simulation.run_simulation].

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
