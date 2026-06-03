# Bubble models

A complete bubble model is a combination of:

1. An **equation of motion** — the ODE governing $R(t)$.
2. A **gas model** — the pressure law inside the bubble.
3. A **shell model** — surface tension and shell stresses.
4. A **medium model** — viscous/elastic stresses in the surrounding liquid.

All components implement a common interface and can be combined freely. The EoM calls the gas, shell, and medium models to compute the liquid-side boundary pressure:

$$p_L = p_\text{gas} - p_\text{shell} - p_\text{medium}$$

and uses $p_L$ to form the ODE right-hand side. Derivatives of $p_L$ are computed automatically via `jax.grad` — no hand-coded Jacobians required.

---

## Equations of motion

All EoMs share the same constructor signature:

```python
eom = SomeEoM(
    gas=...,
    shell=...,
    medium=...,
    R0=2e-6,
    P_amb=101325.0,
    rho_L=998.0,
    # EoM-specific fields below
)
```

### Rayleigh–Plesset

The classic model for a spherical bubble in an incompressible fluid. Suitable for low driving pressures where liquid compressibility is negligible.

$$R\ddot{R} + \frac{3}{2}\dot{R}^2 = \frac{1}{\rho_L}\left(p_L - P_\infty - p_{ac}(t)\right)$$

```python
from jbubble.bubble.eom import RayleighPlesset
eom = RayleighPlesset(gas=..., shell=..., medium=..., R0=2e-6, P_amb=101325, rho_L=998)
```

### Modified Rayleigh–Plesset

Adds a first-order compressibility correction to the RP equation via the acoustic radiation damping term $\frac{R}{c_L}\dot{p}_\text{gas}$. It is simply the Keller-Miksis equation without the $$O(\dot R/c)$$ prefactor terms. Suitable for low driving pressures and for instances where $$M = \dot R/ c\ll 1$$. 


$$(1 - M)\, R \ddot{R} + \frac{3}{2}\left(1 - \frac{M}{3}\right)\dot{R}^2
= \frac{1}{\rho}(1 + M)\left(p_L - P_{\text{amb}} - p_{\text{ac}}\right)
+ \frac{R}{\rho c}\left(\frac{d p_L}{dt} - \frac{d p_{\text{ac}}}{dt}\right)$$

```python
from jbubble.bubble.eom import ModifiedRayleighPlesset
eom = ModifiedRayleighPlesset(..., c_L=1500.0)
```

### Keller–Miksis

First-order compressible model. The standard choice for moderate-to-high driving pressures. Breaks down when $$M = \dot R/c \approx 1$$.

$$\left(1 - \frac{\dot{R}}{c_L}\right)R\ddot{R} + \frac{3}{2}\left(1 - \frac{\dot{R}}{3c_L}\right)\dot{R}^2 = \left(1 + \frac{\dot{R}}{c_L}\right)\frac{p_L - P_\infty - p_{ac}}{\rho_L} + \frac{R}{\rho_L c_L}\frac{d}{dt}(p_L - p_{ac})$$

The time derivative of $p_L$ is obtained via `jax.grad` applied through the entire gas+shell+medium computation.

```python
from jbubble.bubble.eom import KellerMiksis
eom = KellerMiksis(..., c_L=1500.0)
```

### Gilmore

First-order compressible model. Uses enthalpy $H$ rather than pressure and assumes that the speed of sound $C$ varies with $H$. Handles cases where $$M \approx 1$$ well by suppressing the Mach number during violent collapses. Uses the Tait equation of state for the liquid, from which the enthalpy $H$ and local sound speed $C$ are computed; $\dot{H}$ is expanded analytically via the chain rule.

$$
\left(1 - \frac{\dot{R}}{C}\right) R \ddot{R}
+ \frac{3}{2}\left(1 - \frac{\dot{R}}{3C}\right)\dot{R}^2
= \left(1 + \frac{\dot{R}}{C}\right) H
+ \frac{R}{C}\left(1 - \frac{\dot{R}}{C}\right)\dot{H}
$$

```python
from jbubble.bubble.eom import Gilmore
eom = Gilmore(
    ...,
    n_tait=7.0,       # Tait exponent (water default)
    B_tait=304.9e6,   # Tait constant [Pa] (water default)
)
```

### Leighton tube

!!! warning "Work in progress"
    The confinement models (`LeightonTube` and `SphericalConfinement`) are
    not yet validated against reference solutions. Treat their output as
    indicative only. Known caveats: the Leighton geometry factor `beta` is
    unverified, and `SphericalConfinement` assumes a Newtonian lumen liquid
    (elastic / non-Newtonian medium contributions are ignored — see below).

Rayleigh–Plesset modified for a bubble centred in a rigid cylindrical tube. The tube geometry adds an inertia correction and additional added-mass terms.

$$
R \ddot{R}\left(1 + \frac{R}{\Gamma}\beta\right)
+ \frac{3}{2}\dot{R}^2\left(1 + \frac{4R}{3\Gamma}\beta\right)
= \frac{1}{\rho}\left(p_{L,\text{damped}} - P_{\text{amb}} - p_{\text{ac}}\right)
$$


```python
from jbubble.bubble.eom import LeightonTube
eom = LeightonTube(
    ..., c_L=1500.0,
    tube_radius=1e-3,   # tube inner radius [m]
    tube_length=5e-2,   # tube length [m]
)
```

### Spherical confinement

Coupled two-DOF model: bubble radius $R$ and vessel wall radius $a$. The vessel wall is modelled as a thin elastic shell; surrounding tissue contributes inertia. The $2\times2$ coupled system (liquid continuity + radial momentum) is solved via Cramer's rule. The lumen is assumed Newtonian, so only `medium.mu` is used — elastic / non-Newtonian medium terms are not included.

```python
from jbubble.bubble.eom import SphericalConfinement
eom = SphericalConfinement(
    ..., c_L=1500.0,
    vessel_radius=50e-6,  # vessel inner radius [m]
    vessel_rho=1050.0,    # wall material density [kg/m³]
    vessel_E=1e6,         # Young's modulus [Pa]
    vessel_nu=0.49,       # Poisson's ratio
    vessel_d=1e-6,        # wall thickness [m]
    tissue_rho=1050.0,
    tissue_d=1e-3,
)
```

`SphericalConfinement` integrates a `ConfinedBubbleState`, which carries both $R$ and $a$ as ODE variables.

---

## Gas models

### PolytropicGas

The most common gas model. Assumes a polytropic process:

$$p_\text{gas} = P_{\text{gas},0} \left(\frac{R_0}{R}\right)^{3\gamma}$$.

In an isothermal process, heat transfer is fast compared bubble oscillations and $\gamma = 1$ for all gases. 

In an adiabatic process, heat transfer is slow compared to bubble oscillations. The adiabatic index for SF6, which is commonly used as the gaseous core of lipid-coated microbubbles, is $$approx 1.095$$. 


```python
from jbubble.bubble.gas import PolytropicGas
gas = PolytropicGas(gamma=1.095)
```

`gamma` accepts a plain float or any `Property` — useful for learned or state-dependent $\gamma$.

### VanDerWaalsGas

Adds a hard-core repulsion to the polytropic gas model: the bubble cannot be compressed below a fraction $h = h_\text{frac} \cdot R_0$ of its equilibrium radius. Relevant for highly driven bubbles near minimum radius.

$$p_\text{gas} = P_{\text{gas},0} \left(\frac{R_0^3 - h^3}{R^3 - h^3}\right)^\gamma$$

```python
from jbubble.bubble.gas import VanDerWaalsGas
gas = VanDerWaalsGas(gamma=1.4, h_frac=0.2)
```

---

## Shell models

### NoShell

No coating — bare bubble in a liquid. The bubble's surface contributes to the interfacial stresses through the Laplace pressure ($2\sigma/R$), whereby $\sigma$ is the constant surface tension of the surrounding liquid. 

```python
from jbubble.bubble.shell import NoShell
shell = NoShell(sigma=0.072)  # water–air surface tension [N/m]
```

### LipidShell

The effects of a thin lipid monolayer is captured by a radius-dependent surface tension law, specifically the piecewise Marmottant model or its differentiable form, the Marmottant-Gompertz law. The lipid coating contributes to the interfacial stresses through 

- the Laplace pressure  ($2\sigma(R)/R$)
- the lipid's surface dilatational viscosity, which adds the viscous stress term $p_\text{viscous} = \frac{4\kappa_s \dot{R}}{R^2}$



The effective surface tension is encoded as a `Property` (e.g. `MarmottantSurfaceTension` or `GompertzSurfaceTension`) passed to `sigma`.

```python
from jbubble.bubble.shell import LipidShell
from jbubble.bubble.shell import GompertzSurfaceTension

sigma = GompertzSurfaceTension(
    R_buckle_ratio=0.98,  # buckling radius as fraction of R0
    chi=0.55,             # shell elasticity [N/m]
    sigma_rupture=0.072,  # post-rupture value [N/m]
)
shell = LipidShell(sigma=sigma, kappa_s=2.4e-9)  # kappa_s [N·s/m]
```

!!! tip "Use GompertzSurfaceTension for fitting"
    `MarmottantSurfaceTension` is piecewise and has discontinuous derivatives at the phase boundaries. `GompertzSurfaceTension` is a smooth $C^\infty$ approximation that matches the three-regime behaviour and is strongly preferred for gradient-based fitting.

### ThickShell

Church (1995) thick viscoelastic shell. Models polymer-shelled agents (e.g. PLGA). The shell has finite thickness $d_s$ and both elastic ($G_s$) and viscous ($\mu_s$) stiffness:

$$p_\text{elastic} = \frac{4}{3}G_s \frac{d_s}{R_0}\left(1 - \left(\frac{R_0}{R}\right)^3\right)$$
$$p_\text{viscous} = \frac{4\mu_s d_s \dot{R}}{R^2}$$

```python
from jbubble.bubble.shell import ThickShell
shell = ThickShell(
    sigma=0.04,    # surface tension [N/m]
    d_s=15e-9,     # shell thickness [m]
    G_s=10e6,      # shear modulus [Pa]
    mu_s=0.5,      # shell viscosity [Pa·s]
)
```

---

## Medium models


### NewtonianMedium 

Dynamic viscosity $\mu$ is constant and independent of fluid flow. Contributes to the interfacial stresses through 

$$p_\text{viscous} = \frac{4\mu \dot{R}}{R}$$

```python
from jbubble.bubble.medium import NewtonianMedium
medium = NewtonianMedium(mu=1e-3)  # [Pa·s]
```

### KelvinVoigtMedium

Linear viscoelastic medium (solid or gel). Contributes to the interfacial stresses through

- a viscosity term $$p_\text{viscous} = \frac{4\mu \dot{R}}{R}$$
- a spring-like elastic restoring term $$p_\text{elastic} = \frac{4G}{3}\left[\left(\frac{R}{R_0}\right)^3 - 1\right]$$

Valid only for small strains ($|R - R_0| \ll R_0$). Prefer `NeoHookeanMedium` for large oscillations.

```python
from jbubble.bubble.medium import KelvinVoigtMedium
medium = KelvinVoigtMedium(mu=1e-3, G=1e3)
```

### NeoHookeanMedium

Finite-strain viscoelastic medium. Derived by integrating the neo-Hookean constitutive law from $R$ to $\infty$ through the incompressible surrounding solid:

$$p_\text{elastic} = G\left(\frac{5}{2} - 2\frac{R_0}{R} - \frac{1}{2}\left(\frac{R_0}{R}\right)^4\right)$$

Key properties:
- Zero at $R = R_0$ (no stress at equilibrium).
- Reduces to the Kelvin–Voigt result for small strains.
- Saturates to $5G/2$ as $R \to \infty$ (physical softening as surrounding material spreads thin).
- Diverges to $-\infty$ as $R \to 0$ (strong compression resistance).

**Recommended** over Kelvin–Voigt whenever large amplitude oscillations are expected.

```python
from jbubble.bubble.medium import NeoHookeanMedium
medium = NeoHookeanMedium(mu=1e-3, G=1e3)
```

### PowerLawMedium

Generalised Newtonian (power-law) fluid. The consistency index $K$ (`mu` field) replaces the dynamic viscosity:

$$p_\text{viscous} = \frac{4K}{n}\left(2\left|\frac{\dot{R}}{R}\right|\right)^{n-1}\frac{\dot{R}}{R}$$

- $n < 1$ — shear-thinning (blood, mucus, some polymer solutions)
- $n = 1$ — recovers `NewtonianMedium` exactly
- $n > 1$ — shear-thickening

The $1/n$ prefactor arises from integrating the spatially varying viscosity field $\eta(r)$ over the incompressible flow, not from evaluating at the wall.

```python
from jbubble.bubble.medium import PowerLawMedium
medium = PowerLawMedium(mu=1e-3, n_exp=0.6)  # shear-thinning
```

---

## Choosing models for your application

| Application | EoM | Gas | Shell | Medium |
|---|---|---|---|---|
| Free bubble, low pressure | `RayleighPlesset` | `PolytropicGas` | `NoShell` | `NewtonianMedium` |
| Free bubble, high pressure | `KellerMiksis` | `VanDerWaalsGas` | `NoShell` | `NewtonianMedium` |
| Clinical UCA (SonoVue) | `KellerMiksis` | `PolytropicGas` | `LipidShell` + `Gompertz` | `NewtonianMedium` |
| Polymer UCA (PLGA/Optison) | `KellerMiksis` | `PolytropicGas` | `ThickShell` | `NewtonianMedium` |
| Tissue-embedded bubble | `KellerMiksis` | `PolytropicGas` | `NoShell` | `NeoHookeanMedium` |
| Shear-thinning blood | `KellerMiksis` | `PolytropicGas` | `LipidShell` | `PowerLawMedium` |
| Confined vessel | `SphericalConfinement` | `PolytropicGas` | `LipidShell` | `NewtonianMedium` |
| Gradient-based fitting | any | any | `GompertzSurfaceTension` | any |
