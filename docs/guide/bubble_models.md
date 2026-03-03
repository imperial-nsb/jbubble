# Bubble models

jbubble provides nine bubble models covering a range of physical assumptions.
Choosing the right model depends on whether your bubble has a phospholipid shell,
what medium surrounds it, and whether liquid compressibility or spatial
confinement are important.

## Model overview

| Model | Shell | Surrounding medium | Compressibility | Geometry |
|---|---|---|---|---|
| `RayleighPlesset` | None (bare gas) | Newtonian liquid | Incompressible | Free field |
| `Marmottant` | Piecewise $\sigma(R)$ | Newtonian liquid | 1st-order damping | Free field |
| `MarmottantGompertz` | Smooth Gompertz $\sigma(R)$ | Newtonian liquid | 1st-order damping | Free field |
| `KellerMiksisGompertz` | Smooth Gompertz $\sigma(R)$ | Newtonian liquid | Full Keller--Miksis | Free field |
| `KelvinVoigtGompertz` | Smooth Gompertz $\sigma(R)$ | Kelvin--Voigt viscoelastic solid | None | Free field |
| `NeoHookeanGompertz` | Smooth Gompertz $\sigma(R)$ | Neo-Hookean viscoelastic solid | None | Free field |
| `ChurchGompertz` | Thick shell ($G_s$, $d_s$, $\mu_s$) + Gompertz | Newtonian liquid | None | Free field |
| `LeightonGompertz` | Smooth Gompertz $\sigma(R)$ | Newtonian liquid | 1st-order | Rigid tube |
| `SphericalConfinement` | Smooth Gompertz $\sigma(R)$ | Newtonian liquid | 1st-order | Elastic vessel |

!!! tip "Choosing a model"
    Start with `MarmottantGompertz` for most encapsulated microbubble work. Use
    `KellerMiksisGompertz` when peak radial velocities exceed ~10% of the speed
    of sound in the liquid, `KelvinVoigtGompertz` or `NeoHookeanGompertz` for
    tissue-embedded bubbles (prefer Neo-Hookean at large strains), `ChurchGompertz`
    for albumin-shelled agents, and `LeightonGompertz` or `SphericalConfinement`
    when vessel geometry matters.

## Class hierarchy

```
Bubble (abstract)
├── RayleighPlesset
├── _MarmottantEquation, Bubble
│   └── Marmottant
└── GompertzBubble (abstract; smooth Gompertz surface tension)
    ├── _MarmottantEquation, GompertzBubble
    │   └── MarmottantGompertz
    ├── KelvinVoigtGompertz
    │   └── NeoHookeanGompertz
    ├── KellerMiksisGompertz
    ├── ChurchGompertz
    ├── LeightonGompertz
    └── SphericalConfinement
```

---

## Shared physics

### Equilibrium gas pressure

$$
P_{\text{gas},0} = P_{\text{amb}} + \frac{2\,\sigma(R_0)}{R_0}
$$

### Polytropic gas law

$$
P_{\text{gas}} = P_{\text{gas},0}\!\left(\frac{R_0}{R}\right)^{\!3\gamma}
$$

### Van der Waals gas law (Marmottant variants only)

$$
P_{\text{gas}} = P_{\text{gas},0}\!\left(
  \frac{R_0^3 - b^3}{R^3 - b^3}
\right)^{\!\gamma},
\qquad b = \frac{R_0}{5.61}
$$

### Laplace pressure

$$
P_{\text{Laplace}} = \frac{2\,\sigma}{R}
$$

### Liquid viscous pressure

$$
P_{\mu} = \frac{4\,\mu_L\,\dot{R}}{R}
$$

### Shell surface-dilatational viscous pressure

$$
P_{\kappa} = \frac{4\,\kappa_s\,\dot{R}}{R^2}
$$

### Gompertz surface tension

The piecewise $\sigma(R)$ of the original Marmottant model is not
differentiable at the transition radii, which prevents the use of `jax.grad`
through the bubble ODE. All `*Gompertz` models replace the piecewise law with
a smooth Gompertz sigmoid:

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
| $R_{\text{break}}/R_0$ | 1.1 | -- | Rupture radius ratio |

---

## Model descriptions

### RayleighPlesset

The classical Rayleigh--Plesset equation for an uncoated gas bubble in an
incompressible, Newtonian liquid. Appropriate as a baseline or for bare gas
bubbles. No shell parameters are required.

**Governing equation**

$$
R\,\ddot{R} + \frac{3}{2}\dot{R}^2
= \frac{1}{\rho_L}\Bigl[
  P_{\text{gas}} - \frac{2\sigma_L}{R} - \frac{4\mu_L\dot{R}}{R}
  - P_{\text{drive}}(t) - P_{\text{amb}}
\Bigr]
$$

**Surface tension:** $\sigma(R) = \sigma_L$ (constant).

**Parameters**

| Field | Symbol | Default | Units |
|-------|--------|---------|-------|
| `gamma` | $\gamma$ | 1.4 | -- |
| `mu_L` | $\mu_L$ | $8.9 \times 10^{-4}$ | Pa s |
| `rho_L` | $\rho_L$ | 1000 | kg/m$^3$ |
| `c_L` | $c_L$ | 1498 | m/s |
| `sigma_L` | $\sigma_L$ | $72 \times 10^{-3}$ | N/m |

**References**

- Rayleigh, Lord (1917). "On the pressure developed in a liquid during the
  collapse of a spherical cavity." *Philosophical Magazine*, 34(200), 94--98.
- Plesset, M. S. (1949). "The dynamics of cavitation bubbles." *Journal of
  Applied Mechanics*, 16(3), 277--282.

---

### Marmottant

Extends the Rayleigh--Plesset equation with the piecewise Marmottant shell model,
which defines the effective surface tension $\sigma(R)$ in three regimes: buckled,
elastic, and ruptured. Radiation damping and van der Waals gas correction are
included. This is the reference model for phospholipid-coated contrast agents.

**Governing equation**

$$
R\,\ddot{R} + \frac{3}{2}\dot{R}^2
= \frac{1}{\rho_L}\Bigl[
  P_{\text{gas}}\,\delta_c
  - \frac{2\sigma}{R}
  - \frac{4\mu_L\dot{R}}{R}
  - \frac{4\kappa_s\dot{R}}{R^2}
  - P_{\text{drive}} - P_{\text{amb}}
\Bigr]
$$

with Van der Waals gas pressure and compressibility damping:

$$
P_{\text{gas}} = P_{\text{gas},0}\!\left(
  \frac{R_0^3 - b^3}{R^3 - b^3}
\right)^{\!\gamma}, \qquad
\delta_c = 1 - \frac{3\gamma\,\dot{R}\,R^3}{c_L\,(R^3 - b^3)}
$$

**Surface tension (piecewise)**

$$
\sigma(R) =
\begin{cases}
  0 & R \le R_{\text{buckle}} \\[4pt]
  \chi\!\left(\dfrac{R^2}{R_{\text{buckle}}^2} - 1\right)
    & R_{\text{buckle}} < R < R_{\text{break}} \\[6pt]
  \sigma_L & R \ge R_{\text{break}}
\end{cases}
$$

where $R_{\text{break}} = R_{\text{buckle}}\sqrt{\sigma_L/\chi + 1}$.

**Parameters**

| Field | Symbol | Default | Units |
|-------|--------|---------|-------|
| `gamma` | $\gamma$ | 1.07 | -- |
| `chi` | $\chi$ | 0.38 | N/m |
| `mu_L` | $\mu_L$ | $8.9 \times 10^{-4}$ | Pa s |
| `kappa_s` | $\kappa_s$ | $2.4 \times 10^{-9}$ | N s/m |
| `rho_L` | $\rho_L$ | 1000 | kg/m$^3$ |
| `c_L` | $c_L$ | 1498 | m/s |
| `sigma_L` | $\sigma_L$ | $72 \times 10^{-3}$ | N/m |
| `vdw_divisor` | -- | 5.61 | -- |
| `R_buckle_ratio` | $R_{\text{buckle}}/R_0$ | 0.99 | -- |

**References**

- Marmottant, P., van der Meer, S., Emmer, M., Versluis, M., de Jong, N.,
  Hilgenfeldt, S., & Lohse, D. (2005). "A model for large amplitude
  oscillations of coated bubbles accounting for buckling and rupture."
  *Journal of the Acoustical Society of America*, 118(6), 3499--3505.

---

### MarmottantGompertz

Replaces the piecewise $\sigma(R)$ of the Marmottant model with a smooth
Gompertz sigmoid. The smooth surface tension law is everywhere differentiable,
making this model suitable for gradient-based parameter fitting with
`jax.grad`. The governing equation is otherwise identical.

**Governing equation:** Same as [Marmottant](#marmottant) (shared via
`_MarmottantEquation` mixin).

**Surface tension:** Gompertz (see [Shared physics](#gompertz-surface-tension))
with $\sigma_{\text{break}} = \sigma_L$.

**Parameters**

| Field | Symbol | Default | Units |
|-------|--------|---------|-------|
| `chi` | $\chi$ | 0.38 | N/m |
| `gamma` | $\gamma$ | 1.07 | -- |
| `rho_L` | $\rho_L$ | 1000 | kg/m$^3$ |
| `R_buckle_ratio` | $R_{\text{buckle}}/R_0$ | 0.99 | -- |
| `mu_L` | $\mu_L$ | $8.9 \times 10^{-4}$ | Pa s |
| `kappa_s` | $\kappa_s$ | $2.4 \times 10^{-9}$ | N s/m |
| `c_L` | $c_L$ | 1498 | m/s |
| `sigma_L` | $\sigma_L$ | $72 \times 10^{-3}$ | N/m |
| `vdw_divisor` | -- | 5.61 | -- |

**References**

- Marmottant et al. (2005) -- see [Marmottant](#marmottant).

---

### KelvinVoigtGompertz

Models the bubble inside a Kelvin--Voigt viscoelastic solid (e.g. brain tissue
or a gel phantom). The surrounding material exhibits both elastic restoring
stress and viscous dissipation, with a linear (small-strain) elastic
approximation. The Gompertz shell is retained. Liquid compressibility is
neglected.

**Governing equation**

$$
R\,\ddot{R} + \frac{3}{2}\dot{R}^2
= \frac{1}{\rho_L}\Bigl[
  P_{\text{gas}}
  - \frac{2\sigma}{R}
  - \frac{4\mu_m\dot{R}}{R}
  - \frac{4\kappa_s\dot{R}}{R^2}
  - P_{\text{elastic}}
  - P_{\text{drive}} - P_{\text{amb}}
\Bigr]
$$

with

$$
P_{\text{elastic}} = \frac{4}{3}\,G\,\frac{R^3 - R_0^3}{R_0^3}
$$

**Surface tension:** Gompertz with $\sigma_{\text{break}} = \sigma_m$.

**Parameters**

| Field | Symbol | Default | Units |
|-------|--------|---------|-------|
| `chi` | $\chi$ | 0.38 | N/m |
| `gamma` | $\gamma$ | 1.07 | -- |
| `rho_L` | $\rho_L$ | 1000 | kg/m$^3$ |
| `R_buckle_ratio` | $R_{\text{buckle}}/R_0$ | 0.99 | -- |
| `mu_m` | $\mu_m$ | $8.9 \times 10^{-4}$ | Pa s |
| `kappa_s` | $\kappa_s$ | $2.4 \times 10^{-9}$ | N s/m |
| `sigma_m` | $\sigma_m$ | $72 \times 10^{-3}$ | N/m |
| `G` | $G$ | 0 | Pa |

**References**

- Yang, X. & Church, C. C. (2005). "A model for the dynamics of gas bubbles
  in soft tissue." *Journal of the Acoustical Society of America*, 118(6),
  3595--3606.

---

### NeoHookeanGompertz

Identical to the Kelvin--Voigt model except the elastic pressure uses the
Neo-Hookean finite-strain constitutive law. This correctly captures large
oscillation amplitudes where the linear Kelvin--Voigt approximation breaks
down. Inherits all other physics from `KelvinVoigtGompertz`.

**Governing equation:** Same as [KelvinVoigtGompertz](#kelvinvoigtgompertz),
replacing the elastic term:

$$
P_{\text{elastic}} = \frac{4}{3}\,G\!\left[
  \left(\frac{R_0}{R}\right)^{\!3} - \left(\frac{R}{R_0}\right)^{\!3}
\right]
$$

**Surface tension:** Gompertz with $\sigma_{\text{break}} = \sigma_m$ (inherited).

**Parameters:** Same as [KelvinVoigtGompertz](#kelvinvoigtgompertz) (all inherited).

**References**

- Yang, X. & Church, C. C. (2005) -- see [KelvinVoigtGompertz](#kelvinvoigtgompertz).
- Gaudron, R., Warnez, M., & Johnsen, E. (2015). "Bubble dynamics in a
  viscoelastic medium with nonlinear elasticity." *Journal of Fluid
  Mechanics*, 766, 54--75.

---

### KellerMiksisGompertz

Like `MarmottantGompertz` but uses the full Keller--Miksis formulation for
acoustic radiation rather than the approximate correction used in `Marmottant`.
Accounts for acoustic radiation losses via time derivatives of the pressure
terms (computed with `jax.grad`). Recommended when the Mach number of the
bubble wall is non-negligible (strongly driven bubbles or low-viscosity host
media).

**Governing equation**

$$
R\,\ddot{R}\!\left(1 - \frac{\dot{R}}{c_L}\right)
+ \frac{3}{2}\dot{R}^2\!\left(1 - \frac{\dot{R}}{3c_L}\right)
= \frac{1}{\rho_L}\!\left(1 + \frac{\dot{R}}{c_L}\right)\!P_{\text{net}}
+ \frac{R}{\rho_L c_L}\dot{P}_{\text{net}}
$$

where

$$
P_{\text{net}} = P_{\text{gas}}
  - \frac{2\sigma}{R}
  - \frac{4\mu_L\dot{R}}{R}
  - \frac{4\kappa_s\dot{R}}{R^2}
  - P_{\text{drive}} - P_{\text{amb}}
$$

The implementation rearranges this into the explicit form
$\ddot{R} = D / C$ where $C$ and $D$ collect the $\dot{R}$-dependent
coefficients and time-derivative contributions.

**Surface tension:** Gompertz with $\sigma_{\text{break}} = \sigma_L$.

**Parameters**

| Field | Symbol | Default | Units |
|-------|--------|---------|-------|
| `chi` | $\chi$ | 0.38 | N/m |
| `gamma` | $\gamma$ | 1.07 | -- |
| `rho_L` | $\rho_L$ | 1000 | kg/m$^3$ |
| `R_buckle_ratio` | $R_{\text{buckle}}/R_0$ | 0.99 | -- |
| `mu_L` | $\mu_L$ | $8.9 \times 10^{-4}$ | Pa s |
| `kappa_s` | $\kappa_s$ | $2.4 \times 10^{-9}$ | N s/m |
| `c_L` | $c_L$ | 1498 | m/s |
| `sigma_L` | $\sigma_L$ | $72 \times 10^{-3}$ | N/m |

**References**

- Keller, J. B. & Miksis, M. (1980). "Bubble oscillations of large
  amplitude." *Journal of the Acoustical Society of America*, 68(2),
  628--633.

---

### ChurchGompertz

Thick-shell model following Church (1995). The shell has finite thickness
$d_s$, shear modulus $G_s$, and viscosity $\mu_s$, parametrised directly
rather than through effective surface quantities. The shell contributions
scale with $d_s / R_0$.

**Governing equation**

$$
R\,\ddot{R} + \frac{3}{2}\dot{R}^2
= \frac{1}{\rho_L}\Bigl[
  P_{\text{gas}}
  - \frac{2\sigma}{R}
  - \frac{4\mu_L\dot{R}}{R}
  - P_{\text{elastic}}
  - P_{\text{shell,visc}}
  - P_{\text{drive}} - P_{\text{amb}}
\Bigr]
$$

with Church's thick-shell terms:

$$
P_{\text{elastic}} = \frac{4}{3}\,G_s\,\frac{d_s}{R_0}\!\left[
  1 - \left(\frac{R_0}{R}\right)^{\!3}
\right], \qquad
P_{\text{shell,visc}} = \frac{4\,\mu_s\,d_s\,\dot{R}}{R^2}
$$

**Surface tension:** Gompertz with $\sigma_{\text{break}} = \sigma_L$.

**Parameters**

| Field | Symbol | Default | Units |
|-------|--------|---------|-------|
| `chi` | $\chi$ | 0.38 | N/m |
| `gamma` | $\gamma$ | 1.07 | -- |
| `rho_L` | $\rho_L$ | 1000 | kg/m$^3$ |
| `R_buckle_ratio` | $R_{\text{buckle}}/R_0$ | 0.99 | -- |
| `mu_L` | $\mu_L$ | $8.9 \times 10^{-4}$ | Pa s |
| `sigma_L` | $\sigma_L$ | $72 \times 10^{-3}$ | N/m |
| `d_s` | $d_s$ | $4 \times 10^{-9}$ | m |
| `G_s` | $G_s$ | $10 \times 10^{6}$ | Pa |
| `mu_s` | $\mu_s$ | 0.5 | Pa s |

**References**

- Church, C. C. (1995). "The effects of an elastic solid surface layer on
  the radial pulsations of gas bubbles." *Journal of the Acoustical Society
  of America*, 97(3), 1510--1521.

---

### LeightonGompertz

Extends `MarmottantGompertz` to account for the presence of a rigid
cylindrical tube (e.g. a vessel or catheter). Tube geometry modifies the
inertial terms on the left-hand side of the Rayleigh--Plesset equation via
aspect-ratio-dependent multipliers $\alpha$ and $\beta$. Uses the Leighton
confinement correction with first-order compressibility. Appropriate when
the bubble-to-vessel size ratio is non-negligible.

**Governing equation**

$$
R\,\ddot{R}\!\left[1 + \frac{R}{\Gamma_1}\beta\right]
+ \frac{3}{2}\dot{R}^2\!\left[1 + \frac{4R}{3\Gamma_1}\beta\right]
= \frac{1}{\rho_L}\Bigl[
  P_{\text{gas}}\,\delta_c
  - \frac{2\sigma}{R}
  - \frac{4\mu_L\dot{R}}{R}
  - \frac{4\kappa_s\dot{R}}{R^2}
  - P_{\text{drive}} - P_{\text{amb}}
\Bigr]
$$

where $\Gamma_1$ is the tube radius, $\zeta_1 = L_{\text{tube}}/2$ is the
half-length, and:

$$
\alpha = \frac{\zeta_1}{\Gamma_1}\!\left(
  1 + \frac{8\Gamma_1}{3\pi\zeta_1}
\right) - 1, \qquad
\beta = 2\alpha, \qquad
\delta_c = 1 - \frac{3\gamma\dot{R}}{c_L}
$$

**Surface tension:** Gompertz with $\sigma_{\text{break}} = \sigma_L$.

**Parameters**

| Field | Symbol | Default | Units |
|-------|--------|---------|-------|
| `chi` | $\chi$ | 0.38 | N/m |
| `gamma` | $\gamma$ | 1.07 | -- |
| `rho_L` | $\rho_L$ | 1000 | kg/m$^3$ |
| `R_buckle_ratio` | $R_{\text{buckle}}/R_0$ | 0.99 | -- |
| `mu_L` | $\mu_L$ | $8.9 \times 10^{-4}$ | Pa s |
| `kappa_s` | $\kappa_s$ | $2.4 \times 10^{-9}$ | N s/m |
| `c_L` | $c_L$ | 1498 | m/s |
| `sigma_L` | $\sigma_L$ | $72 \times 10^{-3}$ | N/m |
| `tube_radius` | $\Gamma_1$ | $10 \times 10^{-6}$ | m |
| `tube_length` | $L_{\text{tube}}$ | $100 \times 10^{-6}$ | m |

**References**

- Leighton, T. G. (2011). "The inertial terms in equations of motion for
  bubbles in tubular vessels or parties of bubbles in general." *Journal of
  the Acoustical Society of America*, 130(5), 3184--3204.

---

### SphericalConfinement

Models a bubble inside an elastic spherical shell (e.g. a compliant blood
vessel). The bubble radius $R$ and vessel inner radius $a$ are coupled via a
$2 \times 2$ linear system, giving rise to two normal modes. The state vector
is four-dimensional: $[R,\,\dot{R},\,a,\,\dot{a}]$.

**Governing equations**

The coupled system is written as:

$$
\begin{pmatrix} A & B \\ C & D \end{pmatrix}
\begin{pmatrix} \ddot{R} \\ \ddot{a} \end{pmatrix}
=
\begin{pmatrix} E \\ F \end{pmatrix}
$$

with

$$
\begin{aligned}
A &= R^2, \quad B = -a^2, \quad
C = \rho_L R^2\!\left(\frac{1}{R} - \frac{1}{a}\right), \quad
D = \rho_v d_v + \rho_t d_t \\[6pt]
E &= 2a\dot{a}^2 - 2R\dot{R}^2 \\[6pt]
F &= P_{\text{gas}}
   - 2R\dot{R}\rho_L\!\left(\frac{1}{R} - \frac{1}{a}\right)
   - \frac{2\sigma}{R}
   - 4\mu_L\!\left(\frac{\dot{R}}{R} + \frac{\dot{a}}{a}\right)
   - \frac{4\kappa_s\dot{R}}{R^2}
   - P_{\text{wall}}
   - P_{\text{amb}} - P_{\text{drive}}
\end{aligned}
$$

where

$$
P_{\text{gas}} = P_{\text{gas},0}\!\left(\frac{R_0}{R}\right)^{\!3\gamma}
\!\left(1 - \frac{3\gamma\dot{R}}{c_L}\right), \qquad
P_{\text{wall}} = \frac{E_v\,(a - a_0)}{(1 - \nu^2)\,a^2}, \quad \nu = 0.5
$$

The accelerations are solved by Cramer's rule:

$$
\ddot{R} = \frac{ED - BF}{\Delta}, \qquad
\ddot{a} = \frac{AF - CE}{\Delta}, \qquad
\Delta = AD - BC
$$

**Surface tension:** Gompertz with $\sigma_{\text{break}} = \sigma_L$.

**Parameters**

| Field | Symbol | Default | Units |
|-------|--------|---------|-------|
| `chi` | $\chi$ | 0.38 | N/m |
| `gamma` | $\gamma$ | 1.07 | -- |
| `rho_L` | $\rho_L$ | 1000 | kg/m$^3$ |
| `R_buckle_ratio` | $R_{\text{buckle}}/R_0$ | 0.99 | -- |
| `mu_L` | $\mu_L$ | $8.9 \times 10^{-4}$ | Pa s |
| `kappa_s` | $\kappa_s$ | $2.4 \times 10^{-9}$ | N s/m |
| `c_L` | $c_L$ | 1498 | m/s |
| `sigma_L` | $\sigma_L$ | $72 \times 10^{-3}$ | N/m |
| `vessel_radius` | $a_0$ | $15 \times 10^{-6}$ | m |
| `vessel_rho` | $\rho_v$ | 900 | kg/m$^3$ |
| `vessel_E` | $E_v$ | $1 \times 10^{6}$ | Pa |
| `vessel_d` | $d_v$ | $1 \times 10^{-6}$ | m |
| `tissue_rho` | $\rho_t$ | 900 | kg/m$^3$ |
| `tissue_d` | $d_t$ | $1 \times 10^{-6}$ | m |

**References**

- Sassaroli, E. & Hynynen, K. (2005). "Resonance frequency of microbubbles
  in small blood vessels: a numerical study." *Physics in Medicine and
  Biology*, 50(22), 5293--5305.
