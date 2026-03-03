# Bubble models

jbubble provides nine bubble models covering a range of physical assumptions.
Choosing the right model depends on whether your bubble has a phospholipid shell,
what medium surrounds it, and whether liquid compressibility or spatial
confinement are important.

## Model overview

| Model | Shell | Surrounding medium | Compressibility | Geometry |
|---|---|---|---|---|
| `RayleighPlesset` | None (bare gas) | Newtonian liquid | Incompressible | Free field |
| `Marmottant` | Piecewise σ(R) | Newtonian liquid | KM-style radiation damping | Free field |
| `MarmottantGompertz` | Smooth Gompertz σ(R) | Newtonian liquid | KM-style radiation damping | Free field |
| `KellerMiksisGompertz` | Smooth Gompertz σ(R) | Newtonian liquid | Full Keller–Miksis | Free field |
| `KelvinVoigtGompertz` | Smooth Gompertz σ(R) | Kelvin–Voigt viscoelastic solid | None | Free field |
| `NeoHookeanGompertz` | Smooth Gompertz σ(R) | Neo-Hookean viscoelastic solid | None | Free field |
| `ChurchGompertz` | Thick shell (G_s, d_s, μ_s) + Gompertz | Newtonian liquid | None | Free field |
| `LeightonGompertz` | Smooth Gompertz σ(R) | Newtonian liquid | 1st-order | Rigid tube |
| `SphericalConfinement` | Smooth Gompertz σ(R) | Newtonian liquid | 1st-order | Elastic vessel |

## Model descriptions

### RayleighPlesset

The classical Rayleigh–Plesset equation for an uncoated gas bubble in an
incompressible, Newtonian liquid. Appropriate as a baseline or for bare gas
bubbles. No shell parameters are required.

### Marmottant

Extends the Rayleigh–Plesset equation with the piecewise Marmottant shell model,
which defines the effective surface tension σ(R) in three regimes: buckled, elastic,
and ruptured. Radiation damping is included via a Keller–Miksis-style correction.
This is the reference model for phospholipid-coated contrast agents.

### MarmottantGompertz

Replaces the piecewise σ(R) of the Marmottant model with a smooth Gompertz
sigmoid. The smooth surface tension law is everywhere differentiable, making
this model suitable for gradient-based parameter fitting with `jax.grad`.

### KellerMiksisGompertz

Like `MarmottantGompertz` but uses the full Keller–Miksis formulation for
acoustic radiation rather than the approximate correction used in `Marmottant`.
Recommended when the Mach number of the bubble wall is non-negligible (strongly
driven bubbles or low-viscosity host media).

### KelvinVoigtGompertz

Models the bubble inside a Kelvin–Voigt viscoelastic solid (e.g. brain tissue
or a gel phantom). The Gompertz shell is retained. Liquid compressibility is
neglected. Useful for studying bubble dynamics in soft tissue where shear
modulus and viscosity of the surrounding medium are significant.

### LeightonGompertz

Extends `MarmottantGompertz` to account for the presence of a rigid cylindrical
tube (e.g. a vessel or catheter). Uses the Leighton confinement correction with
first-order compressibility. Appropriate when the bubble-to-vessel size ratio is
non-negligible.

### SphericalConfinement

Models a bubble inside an elastic spherical shell (e.g. a compliant blood vessel).
Uses a first-order compressibility correction and the Gompertz surface tension law.
Suitable for studying bubble–vessel wall interaction in microvascular applications.

## The Gompertz surface tension law

The piecewise σ(R) of the original Marmottant model is not differentiable at the
transition radii, which prevents the use of `jax.grad` through the bubble ODE. All
`*Gompertz` models replace the piecewise law with a smooth Gompertz sigmoid:

$$
\sigma(R) = \sigma_{\max} \cdot \exp\!\left(-b \cdot \exp\!\left(-c \cdot \frac{R - R_0}{R_0}\right)\right)
$$

This function closely approximates the Marmottant profile while being
continuously differentiable everywhere, enabling automatic differentiation
through the entire simulation.

!!! tip "Choosing a model"
    Start with `MarmottantGompertz` for most encapsulated microbubble work. Use
    `KellerMiksisGompertz` when peak radial velocities exceed ~10% of the speed
    of sound in the liquid, `KelvinVoigtGompertz` or `NeoHookeanGompertz` for
    tissue-embedded bubbles (prefer Neo-Hookean at large strains), `ChurchGompertz`
    for albumin-shelled agents, and `LeightonGompertz` or `SphericalConfinement`
    when vessel geometry matters.
