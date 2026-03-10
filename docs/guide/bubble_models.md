# Bubble models

jbubble uses a **compositional architecture**: you assemble a bubble model by
choosing independent **gas**, **surface tension**, **shell**, **medium**, and
**equation of motion** components. Any combination of components can be plugged
together — and because the EoMs compute pressure derivatives via `jax.grad`,
everything stays differentiable.

## Architecture overview

```
                        EquationOfMotion
                       ┌────────────────────────┐
                       │  gas:    GasModel       │
  p_L = gas(R)         │  shell:  ShellModel ────┼── sigma: SurfaceTensionModel
      - shell(R, Ṙ)   │  medium: MediumModel    │
      - medium(R, Ṙ)  │  R0, P_amb, rho_L, ...  │
                       └────────────────────────┘
```

The liquid-side boundary pressure is assembled as:

$$
p_L(R, \dot{R}) = p_{\text{gas}}(R)
  - p_{\text{shell}}(R, \dot{R})
  - p_{\text{medium}}(R, \dot{R})
$$

Each equation of motion turns $p_L$ into an ODE right-hand side (e.g. the
Keller–Miksis EoM also differentiates $p_L$ via `jax.grad` to get
$\dot{p}_L$).

## Available components

### Gas models

| Class | Description |
|---|---|
| `PolytropicGas` | $p = P_0 (R_0/R)^{3\gamma}$ — standard polytropic law |
| `VanDerWaalsGas` | $p = P_0 \bigl((R_0^3 - h^3)/(R^3 - h^3)\bigr)^\gamma$ — hard-core corrected |

Both have a `from_equilibrium()` classmethod that computes $P_{\text{gas},0} = P_{\text{amb}} + 2\sigma(R_0)/R_0$ for you.

### Surface tension models

| Class | Description |
|---|---|
| `ConstantSigma` | $\sigma(R) = \sigma_L$ — constant (uncoated bubbles) |
| `MarmottantSigma` | Piecewise Marmottant law (buckled / elastic / ruptured) |
| `GompertzSigma` | Smooth Gompertz sigmoid approximation — **recommended for `jax.grad`** |

`MarmottantSigma` and `GompertzSigma` have `from_R0()` classmethods for easy construction from the equilibrium radius and buckling ratio.

### Shell models

| Class | Description |
|---|---|
| `NoShell` | Laplace pressure only: $2\sigma/R$ |
| `LipidShell` | Thin lipid monolayer: $2\sigma/R + 4\kappa_s \dot{R}/R^2$ |
| `ThickShell` | Church (1995) thick shell: Laplace + elastic + viscous thick-shell terms |

Every `ShellModel` holds a `SurfaceTensionModel` as its `sigma` field.

### Medium models

| Class | Description |
|---|---|
| `KelvinVoigtMedium` | $4\mu \dot{R}/R + \tfrac{4}{3}G\bigl((R/R_0)^3 - 1\bigr)$ — linear viscoelastic. With $G=0$ (default): Newtonian liquid |
| `NeoHookeanMedium` | Finite-strain: $4\mu \dot{R}/R + \tfrac{4}{3}G\bigl((R_0/R)^3 - (R/R_0)^3\bigr)$. Prefer for large oscillation amplitudes |

### Equations of motion

| Class | DOFs | Compressibility | Geometry |
|---|---|---|---|
| `RayleighPlesset` | 2 | None | Free field |
| `ModifiedRayleighPlesset` | 2 | 1st-order gas damping | Free field |
| `KellerMiksis` | 2 | Full 1st-order (autodiff $\dot{p}_L$) | Free field |
| `LeightonTube` | 2 | 1st-order gas damping | Rigid tube |
| `SphericalConfinement` | 4 | 1st-order gas damping | Elastic vessel |

!!! tip "Choosing an EoM"
    Start with `KellerMiksis` for most applications — it correctly handles
    liquid compressibility effects via autodiff and works with any component
    combination. Use `RayleighPlesset` as a baseline for incompressible cases.
    Use `LeightonTube` or `SphericalConfinement` when vessel geometry matters.

---

## Composition examples

### Standard lipid-coated bubble (Keller–Miksis)

```python
from jbubble import (
    PolytropicGas, GompertzSigma, LipidShell, KelvinVoigtMedium, KellerMiksis,
)

R0 = 2.5e-6
sigma  = GompertzSigma.from_R0(R0=R0)
shell  = LipidShell(sigma=sigma)
gas    = PolytropicGas.from_equilibrium(R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0)
medium = KelvinVoigtMedium(R0=R0)

eom = KellerMiksis(
    gas=gas, shell=shell, medium=medium,
    R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0,
)
```

### Uncoated gas bubble (Rayleigh–Plesset)

```python
from jbubble import PolytropicGas, ConstantSigma, NoShell, KelvinVoigtMedium, RayleighPlesset

R0 = 5e-6
sigma  = ConstantSigma()           # constant sigma_L = 0.072 N/m
shell  = NoShell(sigma=sigma)
gas    = PolytropicGas.from_equilibrium(R0=R0, gamma=1.4, P_amb=101325.0, sigma_R0=0.072)
medium = KelvinVoigtMedium(R0=R0)

eom = RayleighPlesset(gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=998.0)
```

### Thick-shell albumin agent (Church model)

```python
from jbubble import PolytropicGas, GompertzSigma, ThickShell, KelvinVoigtMedium, RayleighPlesset

R0 = 2e-6
sigma = GompertzSigma.from_R0(R0=R0)
shell = ThickShell(sigma=sigma, R0=R0, d_s=4e-9, G_s=10e6, mu_s=0.5)
gas   = PolytropicGas.from_equilibrium(R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0)
medium = KelvinVoigtMedium(R0=R0)

eom = RayleighPlesset(gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=998.0)
```

### Tissue-embedded bubble (Kelvin–Voigt)

```python
from jbubble import PolytropicGas, GompertzSigma, LipidShell, KelvinVoigtMedium, RayleighPlesset

R0 = 3e-6
sigma  = GompertzSigma.from_R0(R0=R0)
shell  = LipidShell(sigma=sigma)
gas    = PolytropicGas.from_equilibrium(R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0)
medium = KelvinVoigtMedium(R0=R0, G=10e3)  # G > 0 activates elastic restoring force

eom = RayleighPlesset(gas=gas, shell=shell, medium=medium, R0=R0, P_amb=101325.0, rho_L=998.0)
```

### Neo-Hookean tissue at large strains

```python
from jbubble import PolytropicGas, GompertzSigma, LipidShell, NeoHookeanMedium, KellerMiksis

R0 = 3e-6
sigma  = GompertzSigma.from_R0(R0=R0)
shell  = LipidShell(sigma=sigma)
gas    = PolytropicGas.from_equilibrium(R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0)
medium = NeoHookeanMedium(R0=R0, G=10e3)

eom = KellerMiksis(
    gas=gas, shell=shell, medium=medium,
    R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0,
)
```

### Bubble in a rigid tube (Leighton)

```python
from jbubble import PolytropicGas, GompertzSigma, LipidShell, KelvinVoigtMedium, LeightonTube

R0 = 2.5e-6
sigma  = GompertzSigma.from_R0(R0=R0)
shell  = LipidShell(sigma=sigma)
gas    = PolytropicGas.from_equilibrium(R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0)
medium = KelvinVoigtMedium(R0=R0)

eom = LeightonTube(
    gas=gas, shell=shell, medium=medium,
    R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0,
    tube_radius=10e-6, tube_length=100e-6,
)
```

### Bubble in an elastic vessel (Spherical confinement)

```python
from jbubble import PolytropicGas, GompertzSigma, LipidShell, KelvinVoigtMedium, SphericalConfinement

R0 = 2.5e-6
sigma  = GompertzSigma.from_R0(R0=R0)
shell  = LipidShell(sigma=sigma)
gas    = PolytropicGas.from_equilibrium(R0=R0, gamma=1.07, P_amb=101325.0, sigma_R0=sigma.sigma_R0)
medium = KelvinVoigtMedium(R0=R0)

eom = SphericalConfinement(
    gas=gas, shell=shell, medium=medium,
    R0=R0, P_amb=101325.0, rho_L=998.0, c_L=1481.0,
    vessel_radius=15e-6, vessel_rho=900.0,
    vessel_E=1e6, vessel_d=1e-6,
    tissue_rho=900.0, tissue_d=1e-6,
)
```

The `SphericalConfinement` EoM has a 4-DOF state vector $[R, \dot{R}, a, \dot{a}]$
where $a$ is the vessel wall radius. The coupled system is solved via Cramer's
rule at each time step.

---

## Physics reference

### Equilibrium gas pressure

$$
P_{\text{gas},0} = P_{\text{amb}} + \frac{2\,\sigma(R_0)}{R_0}
$$

### Polytropic gas law

$$
P_{\text{gas}} = P_{\text{gas},0}\!\left(\frac{R_0}{R}\right)^{\!3\gamma}
$$

### Van der Waals gas law

$$
P_{\text{gas}} = P_{\text{gas},0}\!\left(
  \frac{R_0^3 - h^3}{R^3 - h^3}
\right)^{\!\gamma},
\qquad h = \frac{R_0}{5.61}
$$

### Gompertz surface tension

The smooth Gompertz function approximates the piecewise Marmottant surface
tension law, enabling `jax.grad` through the bubble ODE:

$$
\sigma(R) = a\,\exp\!\bigl(-b\,\exp\!\bigl(c\,(1 - R/R_{\text{buckle}})\bigr)\bigr)
$$

with

$$
a = \sigma_{\text{break}}, \qquad
c = \frac{2\chi}{a}\sqrt{1 + \frac{a}{2\chi}}, \qquad
b = \frac{-\ln(\sigma_{R_0}/a)}{\exp\!\bigl(c\,(1-R_0/R_{\text{buckle}})\bigr)}
$$

Boundary behaviour:

| Limit | Value |
|-------|-------|
| $R \to 0$ (buckled) | $\sigma \to 0$ |
| $R = R_0$ (equilibrium) | $\sigma = \sigma_{R_0} = \chi\bigl((R_0/R_{\text{buckle}})^2 - 1\bigr)$ |
| $R \to \infty$ (ruptured) | $\sigma \to \sigma_{\text{break}}$ |

### Piecewise Marmottant surface tension

$$
\sigma(R) =
\begin{cases}
  0 & R \le R_{\text{buckle}} \\[4pt]
  \chi\!\left(\dfrac{R^2}{R_{\text{buckle}}^2} - 1\right)
    & R_{\text{buckle}} < R < R_{\text{break}} \\[6pt]
  \sigma_L & R \ge R_{\text{break}}
\end{cases}
$$

!!! warning
    The piecewise $\sigma(R)$ has discontinuous derivatives at the regime
    boundaries. Use `GompertzSigma` for gradient-based workflows.

### Default physical constants

| Symbol | Value | Units | Description |
|--------|-------|-------|-------------|
| $\gamma_{\text{lipid}}$ | 1.07 | -- | Polytropic exponent (lipid shell) |
| $\gamma_{\text{air}}$ | 1.4 | -- | Polytropic exponent (air) |
| $\chi$ | 0.38 | N/m | Shell elasticity modulus |
| $\kappa_s$ | $2.4 \times 10^{-9}$ | N s/m | Shell surface-dilatational viscosity |
| $\mu_L$ | $8.9 \times 10^{-4}$ | Pa s | Dynamic viscosity (water) |
| $\rho_L$ | 1000 | kg/m$^3$ | Liquid density (water) |
| $c_L$ | 1498 | m/s | Speed of sound (water) |
| $P_{\text{amb}}$ | $1.013 \times 10^5$ | Pa | Atmospheric pressure |
| $\sigma_L$ | $72 \times 10^{-3}$ | N/m | Surface tension (air--water) |
| $R_{\text{buckle}}/R_0$ | 0.99 | -- | Buckling radius ratio |

---

## EoM formulations

### Rayleigh–Plesset

$$
R\,\ddot{R} + \frac{3}{2}\dot{R}^2
= \frac{1}{\rho_L}\bigl[
  p_L - P_{\text{amb}} - p_{\text{ac}}(t)
\bigr]
$$

### Modified Rayleigh–Plesset

Adds first-order gas radiation damping:

$$
R\,\ddot{R} + \frac{3}{2}\dot{R}^2
= \frac{1}{\rho_L}\bigl[
  p_L + \frac{R}{c_L}\dot{p}_{\text{gas}}
  - P_{\text{amb}} - p_{\text{ac}}
\bigr]
$$

where $\dot{p}_{\text{gas}}$ is computed via `jax.grad`.

### Keller–Miksis

Full first-order compressibility on all of $p_L$:

$$
R\,\ddot{R}\!\left(1 - \frac{\dot{R}}{c_L}\right)
+ \frac{3}{2}\dot{R}^2\!\left(1 - \frac{\dot{R}}{3c_L}\right)
= \frac{1}{\rho_L}\!\left(1 + \frac{\dot{R}}{c_L}\right)\!P_{\text{net}}
+ \frac{R}{\rho_L c_L}\dot{P}_{\text{net}}
$$

The key architectural win: $\dot{p}_L$ is computed by `jax.grad(self.p_L, ...)`,
so any combination of gas, shell, and medium components works automatically.

### Leighton tube confinement

Modified inertia terms for a rigid tube of radius $\Gamma$ and length $L$:

$$
R\,\ddot{R}\!\left[1 + \frac{R}{\Gamma}\beta\right]
+ \frac{3}{2}\dot{R}^2\!\left[1 + \frac{4R}{3\Gamma}\beta\right]
= \frac{1}{\rho_L}\bigl[
  p_{L, \text{damped}} - P_{\text{amb}} - p_{\text{ac}}
\bigr]
$$

with $\alpha = (\zeta/\Gamma)(1 + 8\Gamma/(3\pi\zeta)) - 1$, $\beta = 2\alpha$, $\zeta = L/2$.

### Spherical confinement

Coupled 2×2 system for bubble radius $R$ and vessel radius $a$, solved via Cramer's rule. See the API docs for full equations.

---

## References

- Rayleigh (1917). *Phil. Mag.*, 34(200), 94–98.
- Plesset (1949). *J. Appl. Mech.*, 16(3), 277–282.
- Marmottant et al. (2005). *J. Acoust. Soc. Am.*, 118(6), 3499–3505.
- Keller & Miksis (1980). *J. Acoust. Soc. Am.*, 68(2), 628–633.
- Church (1995). *J. Acoust. Soc. Am.*, 97(3), 1510–1521.
- Yang & Church (2005). *J. Acoust. Soc. Am.*, 118(6), 3595–3606.
- Gaudron, Warnez & Johnsen (2015). *J. Fluid Mech.*, 766, 54–75.
- Leighton (2011). *J. Acoust. Soc. Am.*, 130(5), 3184–3204.
- Sassaroli & Hynynen (2005). *Phys. Med. Biol.*, 50(22), 5293–5305.
