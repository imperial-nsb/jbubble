# Bubble models

A bubble model in jbubble has four parts:

1. An **equation of motion** (EoM): the ordinary differential equation for
   the radius $R(t)$.
2. A **gas model**: the pressure inside the bubble.
3. A **shell model**: surface tension and the stresses of a coating.
4. A **medium model**: the viscous and elastic stresses of the surrounding
   liquid or tissue.

The equation of motion asks the other three for the liquid pressure at the
bubble wall,

$$
p_L = p_\text{gas} - p_\text{shell} - p_\text{medium},
$$

and relates it to the wall acceleration $\ddot R$. Where an equation needs a
time derivative, such as $\mathrm{d}p_L/\mathrm{d}t$ in Keller-Miksis, JAX
computes it by automatic differentiation. That's why any gas works with any
shell, medium, and equation of motion, including models that you write
yourself.

Every code block on this page runs as written, in order. The first block
defines the parts that the later ones reuse:

```python
import jax
import jax.numpy as jnp

from jbubble import run_simulation
from jbubble.bubble.eom import KellerMiksis, ModifiedRayleighPlesset, RayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine

water = {"P_amb": 101325.0, "rho_L": 998.0}  # [Pa], [kg/m³]
parts = {
    "gas": PolytropicGas(gamma=1.4),
    "shell": NoShell(sigma=0.072),  # [N/m]
    "medium": NewtonianMedium(mu=1e-3),  # [Pa s]
    "R0": 2e-6,  # [m]
}
```

## Equations of motion

Every equation of motion takes `gas`, `shell`, `medium`, `R0`, `P_amb`, and
`rho_L`. Some take more:

| Equation of motion | Extra parameters | Use it for |
|---|---|---|
| [`RayleighPlesset`][jbubble.bubble.eom.RayleighPlesset] | none | Weak driving in an incompressible liquid |
| [`ModifiedRayleighPlesset`][jbubble.bubble.eom.ModifiedRayleighPlesset] | `c_L` | Coated bubbles at low Mach number, as in Marmottant et al. (2005) |
| [`KellerMiksis`][jbubble.bubble.eom.KellerMiksis] | `c_L` | Most work: the default of every preset |

### Rayleigh-Plesset

The classic equation for a spherical bubble in an incompressible liquid,
where $p_\text{ac}(t)$ is the driving pressure:

$$
R\ddot{R} + \frac{3}{2}\dot{R}^2
    = \frac{1}{\rho_L}\left(p_L - P_\text{amb} - p_\text{ac}\right).
$$

### Modified Rayleigh-Plesset

Adds the radiation damping of the gas pressure only, a first-order
compressibility correction:

$$
R\ddot{R} + \frac{3}{2}\dot{R}^2
    = \frac{1}{\rho_L}\left(p_L + \frac{R}{c_L}\frac{\mathrm{d}p_\text{gas}}{\mathrm{d}t}
    - P_\text{amb} - p_\text{ac}\right).
$$

### Keller-Miksis

Accounts for liquid compressibility to first order in the wall Mach number
$M = \dot R / c_L$:

$$
(1 - M) R\ddot{R} + \frac{3}{2}\left(1 - \frac{M}{3}\right)\dot{R}^2
    = \frac{1 + M}{\rho_L}\left(p_L - P_\text{amb} - p_\text{ac}\right)
    + \frac{R}{\rho_L c_L}\left(\frac{\mathrm{d}p_L}{\mathrm{d}t}
    - \frac{\mathrm{d}p_\text{ac}}{\mathrm{d}t}\right).
$$

### Compare the equations

The following code drives the same free bubble at 400 kPa, hard enough for
an inertial collapse, with three equations of motion:

```{.python continuation}
pulse = ToneBurst(freq=1e6, pressure=400e3, shape=Sine(), cycle_num=3)
eoms = {
    "Rayleigh-Plesset": RayleighPlesset(**parts, **water),
    "modified Rayleigh-Plesset": ModifiedRayleighPlesset(**parts, **water, c_L=1500.0),
    "Keller-Miksis": KellerMiksis(**parts, **water, c_L=1500.0),
}
for name, eom in eoms.items():
    result = run_simulation(eom, pulse)
    R = result.radius / eom.R0
    print(f"{name:25s} R_max/R0 = {R.max():.2f}, R_min/R0 = {R.min():.3f}")
```

Rayleigh-Plesset has no radiation damping, so it overestimates the growth and
the collapse. For the full comparison, with plots, see the example
[Equations of motion](../examples/04_equations_of_motion.md).

## Gas models

[`PolytropicGas`][jbubble.bubble.gas.PolytropicGas] is the standard model:

$$
p_\text{gas} = P_{\text{gas},0}\left(\frac{R_0}{R}\right)^{3\gamma}.
$$

The polytropic exponent $\gamma$ is 1 for an isothermal process, where heat
flows faster than the bubble oscillates, and the adiabatic index for an
adiabatic one: 1.4 for air and about 1.095 for the SF6 of lipid-coated
contrast agents.

[`VanDerWaalsGas`][jbubble.bubble.gas.VanDerWaalsGas] adds a hard core of
radius $h = h_\text{frac} R_0$, the radius of the gas compressed to its
excluded volume, which stops a violent collapse from reaching $R = 0$:

$$
p_\text{gas} = P_{\text{gas},0}
    \left(\frac{R_0^3 - h^3}{R^3 - h^3}\right)^{\gamma}.
$$

```{.python continuation}
from jbubble.bubble.gas import VanDerWaalsGas

air = PolytropicGas(gamma=1.4)
air_with_core = VanDerWaalsGas(gamma=1.4, h_frac=0.1)  # h = 0.1 R0
```

The hard-core fraction `h_frac` depends on the gas species and its density
at equilibrium.

`P_gas0` isn't a parameter: the equation of motion computes it from the
equilibrium at $R_0$, where the gas pressure balances the ambient pressure
and the Laplace pressure of the shell.

## Shell models

### Uncoated bubbles

[`NoShell`][jbubble.bubble.shell.NoShell] contributes only the Laplace
pressure $2\sigma/R$ of the liquid's surface tension, 0.072 N/m for water.

### Lipid shells

[`LipidShell`][jbubble.bubble.shell.LipidShell] models a lipid monolayer
with a radius-dependent surface tension $\sigma(R)$ and a surface-dilatational
viscosity $\kappa_s$:

$$
p_\text{shell} = \frac{2\sigma(R)}{R} + \frac{4\kappa_s\dot{R}}{R^2}.
$$

The surface tension law carries the shell's elasticity. Three laws share
the parameters `R_buckle_ratio` (the buckling radius as a fraction of
$R_0$), `chi` (the elasticity $\chi$), and `sigma_rupture`:

| Law | What it is | Use it for |
|---|---|---|
| [`MarmottantSurfaceTension`][jbubble.bubble.shell.MarmottantSurfaceTension] | The piecewise law of Marmottant et al. (2005): buckled, elastic, and ruptured regimes | Forward simulations with the exact law |
| [`SmoothMarmottantSurfaceTension`][jbubble.bubble.shell.SmoothMarmottantSurfaceTension] | The Marmottant law with corners rounded over a width set by `smoothing` | Gradients and fitting; the [`lipid_bubble`][jbubble.utils.presets.lipid_bubble] preset |
| [`GompertzSurfaceTension`][jbubble.bubble.shell.GompertzSurfaceTension] | The Marmottant-Gompertz law of Gümmer et al. (2021) | Reproducing that paper |

In the elastic regime, all three follow
$\sigma = \chi\left[(R/R_b)^2 - 1\right]$ with $R_b$ the buckling radius.
The smoothed law stays within `smoothing * ln(2) * sigma_rupture`, about
0.5 mN/m at the defaults, of the piecewise law, and converges to it as
`smoothing` goes to zero. The Gompertz law doesn't converge to the
Marmottant law, so a $\chi$ that you fit with it differs from a Marmottant
$\chi$; its Notes give the size of the difference.

A surface tension law is a [`Property`][jbubble.bubble.property.Property]:
a function of the bubble state. The following code evaluates the three laws
below, at, and above the buckling radius:

```{.python continuation}
from jbubble.bubble.shell import (
    GompertzSurfaceTension,
    LipidShell,
    MarmottantSurfaceTension,
    SmoothMarmottantSurfaceTension,
)
from jbubble.bubble.state import BubbleState

shell_params = {"R_buckle_ratio": 0.98058, "chi": 0.5, "sigma_rupture": 0.072}
laws = {
    "Marmottant": MarmottantSurfaceTension(**shell_params),
    "smoothed": SmoothMarmottantSurfaceTension(**shell_params),
    "Gompertz": GompertzSurfaceTension(**shell_params),
}
for name, law in laws.items():
    values = [law(BubbleState(R=x * 2e-6, R0=2e-6)) for x in (0.97, 1.0, 1.05)]
    print(f"{name:10s}", " ".join(f"{1e3 * v:6.2f}" for v in values), "mN/m")

lipid = LipidShell(sigma=laws["smoothed"], kappa_s=7.5e-9)  # kappa_s [N s/m]
```

### Thick shells

[`ThickShell`][jbubble.bubble.shell.ThickShell] models a polymer or protein
shell of thickness $d_s$, shear modulus $G_s$, and shear viscosity $\mu_s$ as
an incompressible viscoelastic layer (Church 1995):

$$
p_\text{elastic} = \frac{4}{3} G_s
    \left[1 - \left(\frac{R_0}{R}\right)^3\right]\frac{V_s}{R^3 - V_s},
\qquad
p_\text{viscous} = 4 \mu_s \frac{V_s}{R^3 - V_s} \frac{\dot{R}}{R},
$$

where $V_s = R_0^3 - (R_0 - d_s)^3$. For a thin shell, these reduce to the
model of Hoff et al. (2000), which adds a stiffness of $12 G_s d_s / R_0$
per unit radial strain.

```{.python continuation}
from jbubble.bubble.shell import ThickShell

polymer = ThickShell(sigma=0.04, d_s=20e-9, G_s=11.7e6, mu_s=0.45)
```

## Medium models

| Medium | Stress on the wall | Use it for |
|---|---|---|
| [`NewtonianMedium`][jbubble.bubble.medium.NewtonianMedium] | $4\mu\dot R/R$ | Water, saline, blood plasma |
| [`KelvinVoigtMedium`][jbubble.bubble.medium.KelvinVoigtMedium] | adds $\tfrac{4G}{3}\left[1 - (R_0/R)^3\right]$ | Small oscillations in a soft solid or gel (Yang and Church 2005) |
| [`NeoHookeanMedium`][jbubble.bubble.medium.NeoHookeanMedium] | adds $G\left[\tfrac{5}{2} - 2\tfrac{R_0}{R} - \tfrac{1}{2}\left(\tfrac{R_0}{R}\right)^4\right]$ | Large oscillations in tissue |
| [`PowerLawMedium`][jbubble.bubble.medium.PowerLawMedium] | $\tfrac{4K}{n}\dot\gamma_\text{eff}^{\,n-1}\dot R/R$ | Shear-thinning ($n < 1$) or shear-thickening ($n > 1$) liquids |

The two elastic laws agree for small strains, $4G(R - R_0)/R_0$, but only the
neo-Hookean law holds at finite strain. For a power-law liquid, `mu` is the
consistency index $K$ in Pa sⁿ, and `eps` regularises the shear rate where
$\dot R$ changes sign.

```{.python continuation}
from jbubble.bubble.medium import NeoHookeanMedium, PowerLawMedium

tissue = NeoHookeanMedium(mu=5e-3, G=10e3)  # [Pa s], [Pa]
shear_thinning = PowerLawMedium(mu=1.5e-2, n_exp=0.7)  # K [Pa s^n]
```

## Properties: constant, state-dependent, or learned

Every physical coefficient of a gas, shell, or medium model, such as `sigma`,
`kappa_s`, `G`, or `gamma`, is a [`Property`][jbubble.bubble.property.Property]:
a function from the bubble state to a scalar. A plain number becomes a
[`ConstantProperty`][jbubble.bubble.property.ConstantProperty]. Pass any
other `Property` to make the coefficient depend on the state, for example a
shear modulus that stiffens with strain:

```{.python continuation}
from jbubble.bubble.medium import KelvinVoigtMedium
from jbubble.bubble.property import Property


class StrainStiffening(Property):
    """Shear modulus G0 (1 + alpha ε²) at the radial strain ε = R/R0 - 1."""

    G0: float  # [Pa]
    alpha: float

    def __call__(self, state):
        strain = state.R / state.R0 - 1.0
        return self.G0 * (1.0 + self.alpha * strain**2)


gel = KelvinVoigtMedium(mu=1e-3, G=StrainStiffening(G0=5e3, alpha=20.0))
```

[`NeuralProperty`][jbubble.bubble.property.NeuralProperty] wraps a neural
network that maps $R/R_0$ to the value. Bound its output with the network's
final activation, for example to keep a surface tension between 0 and the
rupture value:

```{.python continuation}
import equinox as eqx

from jbubble.bubble.property import NeuralProperty

net = eqx.nn.MLP(
    in_size=1,
    out_size=1,
    width_size=16,
    depth=2,
    final_activation=lambda x: 0.072 * jax.nn.sigmoid(x),  # [N/m]
    key=jax.random.key(0),
)
learned_shell = LipidShell(sigma=NeuralProperty(net=net), kappa_s=7.5e-9)
```

To fit such a network to data, see
[Learn a constitutive law with a neural network](fitting.md#learn-a-constitutive-law-with-a-neural-network).
To write a new medium or shell model rather than a coefficient, subclass
[`MediumModel`][jbubble.bubble.medium.MediumModel] or
[`ShellModel`][jbubble.bubble.shell.ShellModel]; the example
[Custom physics](../examples/08_custom_physics.md) shows how.

## Set the initial state

By default, a simulation starts at rest at the equilibrium radius,
[`eom.initial_state()`][jbubble.bubble.eom.EquationOfMotion.initial_state].
To start elsewhere, pass `state0` to `run_simulation`. The following code
releases a bubble at rest from 1.5 times its equilibrium radius, with no
drive:

```{.python continuation}
eom = KellerMiksis(**parts, **water, c_L=1500.0)
silence = ToneBurst(freq=1e6, pressure=0.0, shape=Sine(), cycle_num=5)
state0 = eom.initial_state(R=1.5 * eom.R0)  # R_dot defaults to 0
result = run_simulation(eom, silence, state0=state0)
print(f"R_min/R0 after release: {result.radius.min() / eom.R0:.3f}")
```

`state0` holds the radius `R`, the wall velocity `R_dot`, and the
equilibrium values `R0` and `P_gas0`, which stay constant during the solve.
`BubbleState(R=1.5 * R0)` gives the same start: a zero `R0` or `P_gas0` means
"unset", and the solver fills it from the equation of motion. For an empty
cavity, set `P_gas0` to a tiny positive value, such as `1e-12`.

## Choose models for your application

| Application | Equation of motion | Gas | Shell | Medium |
|---|---|---|---|---|
| Free bubble, weak driving | `KellerMiksis` or `RayleighPlesset` | `PolytropicGas` | `NoShell` | `NewtonianMedium` |
| Inertial cavitation | `KellerMiksis` | `VanDerWaalsGas` | `NoShell` | `NewtonianMedium` |
| Lipid-coated contrast agent, such as SonoVue | `KellerMiksis` | `PolytropicGas` | `LipidShell` with `SmoothMarmottantSurfaceTension` | `NewtonianMedium` |
| Polymer- or protein-shelled agent | `KellerMiksis` | `PolytropicGas` | `ThickShell` | `NewtonianMedium` |
| Bubble in tissue | `KellerMiksis` | `PolytropicGas` | any | `NeoHookeanMedium` |
| Non-Newtonian liquid | `KellerMiksis` | `PolytropicGas` | any | `PowerLawMedium` |

The presets [`free_bubble`][jbubble.utils.presets.free_bubble],
[`lipid_bubble`][jbubble.utils.presets.lipid_bubble], and
[`thick_shell_bubble`][jbubble.utils.presets.thick_shell_bubble] assemble the
first, third, and fourth rows with cited parameters. Small or
viscous bubbles can make the equations stiff; see
[Solvers and stiffness](solvers.md).
