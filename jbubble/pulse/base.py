"""Core abstractions for acoustic driving pulses."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp


class Envelope(eqx.Module, abc.ABC):
    """Window function mapping relative time *tau* to a scale in [0, 1].

    Called as ``envelope(tau, duration)`` where *tau* = t − initial_time.
    Returns 0 outside [0, duration].
    """

    @abc.abstractmethod
    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
        ...


class RectangularEnvelope(Envelope):
    """Hard on/off gating — 1 inside [0, duration], 0 outside."""

    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
        return jnp.where((tau >= 0) & (tau <= duration), 1.0, 0.0)


class HannEnvelope(Envelope):
    """Hann (raised-cosine) window for smooth on/off transitions."""

    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
        in_window = (tau >= 0) & (tau <= duration)
        hann = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau / duration))
        return jnp.where(in_window, hann, 0.0)


class TukeyEnvelope(Envelope):
    """Tukey (tapered cosine) window — flat in the middle, cosine tapers.

    Parameters
    ----------
    alpha : float
        Fraction of the window inside the cosine tapers.
        ``alpha = 0`` → rectangular, ``alpha = 1`` → Hann.
    """

    alpha: float = 0.5

    def __call__(self, tau: jax.Array, duration: float) -> jax.Array:
        in_window = (tau >= 0) & (tau <= duration)
        frac = tau / duration  # normalised position in [0, 1]

        # Lower taper: 0 <= frac < alpha/2
        lower = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi / self.alpha * (frac - self.alpha / 2.0)))
        # Upper taper: 1 - alpha/2 < frac <= 1
        upper = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi / self.alpha * (frac - 1.0 + self.alpha / 2.0)))

        val = jnp.where(frac < self.alpha / 2.0, lower, 1.0)
        val = jnp.where(frac > 1.0 - self.alpha / 2.0, upper, val)
        return jnp.where(in_window, val, 0.0)


# ---------------------------------------------------------------------------
# Pulse base
# ---------------------------------------------------------------------------


class Pulse(eqx.Module, abc.ABC):
    """Abstract acoustic driving pulse.

    Every ``Pulse`` is callable: ``pulse(t)`` returns the instantaneous
    pressure [Pa] at time ``t``.  Implementations must be JAX-differentiable
    so that equations of motion (e.g. Keller–Miksis) can compute
    ``jax.grad(pulse)(t)``.

    Subclasses implement :meth:`_evaluate` (the raw, un-enveloped signal).
    The base :meth:`__call__` applies the :attr:`envelope` automatically.

    Operator overloads for composition::

        combined = pulse_a + pulse_b          # → Summed
        scaled   = 0.5 * pulse_a              # → Scaled
        windowed = combined.windowed(HannEnvelope())  # swap envelope
    """

    initial_time: float = eqx.field(default=0.0, kw_only=True)
    envelope: Envelope = eqx.field(default_factory=RectangularEnvelope, kw_only=True)

    @abc.abstractmethod
    def _evaluate(self, t: jax.Array) -> jax.Array:
        """Raw signal value at time *t*, before envelope application."""
        ...

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """Active pulse duration [s] (excluding any leading silence)."""
        ...

    @property
    def t_end(self) -> float:
        """Suggested simulation end time [s].

        Default: ``initial_time + 2 × duration``.
        """
        return self.initial_time + 2.0 * self.duration

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate pressure at time *t* [Pa], with envelope applied."""
        tau = t - self.initial_time
        return self._evaluate(t) * self.envelope(tau, self.duration)

    def __add__(self, other: "Pulse") -> "Summed":
        """Add another pulse: pulse_a + pulse_b"""
        left = self.pulses if isinstance(self, Summed) else (self,)
        right = other.pulses if isinstance(other, Summed) else (other,)
        return Summed(pulses=left + right)

    def __radd__(self, other: "Pulse") -> "Summed":
        """Right addition: other + self.  If *other* is a Pulse, delegate to its __add__."""
        if isinstance(other, Pulse):
            return other.__add__(self)
        return NotImplemented

    def __mul__(self, factor: float) -> "Scaled":
        """Scale the pulse by a factor: pulse * factor"""
        return Scaled(pulse=self, factor=float(factor))

    def __rmul__(self, factor: float) -> "Scaled":
        """Right multiplication: factor * pulse"""
        return Scaled(pulse=self, factor=float(factor))

    def __neg__(self) -> "Scaled":
        """Negate the pulse (flip polarity): -pulse"""
        return Scaled(pulse=self, factor=-1.0)

    def __pos__(self) -> "Pulse":
        """Unary plus (identity): +pulse"""
        return self

    def __sub__(self, other: "Pulse") -> "Summed":
        """Subtract another pulse: pulse_a - pulse_b"""
        return self + (-other)

    def __rsub__(self, other: "Pulse") -> "Summed":
        """Right subtraction: other - self"""
        if isinstance(other, Pulse):
            return other + (-self)
        return NotImplemented

    def __truediv__(self, factor: float) -> "Scaled":
        """Divide pulse by a factor: pulse / 2.0"""
        return Scaled(pulse=self, factor=1.0 / float(factor))

    def __rtruediv__(self, factor: float):
        """Division by pulse not supported: scalar / pulse"""
        return NotImplemented

    def windowed(self, envelope: Envelope) -> "Pulse":
        """Return a copy of this pulse with *envelope* replacing the current one."""
        return eqx.tree_at(
            lambda p: p.envelope, self, envelope,
            is_leaf=lambda x: isinstance(x, Envelope),
        )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class Scaled(Pulse):
    """Amplitude-scaled version of another pulse.

    ``Scaled`` is transparent: it delegates entirely to the child pulse's
    ``__call__`` (which already applies the child's envelope) and simply
    multiplies by *factor*.  No additional envelope is applied.

    Parameters
    ----------
    pulse : Pulse
        The pulse to scale.
    factor : float
        Multiplicative factor.
    """

    pulse: Pulse
    factor: float

    @property
    def duration(self) -> float:
        return self.pulse.duration

    @property
    def t_end(self) -> float:
        return self.pulse.t_end

    def _evaluate(self, t: jax.Array) -> jax.Array:
        return self.factor * self.pulse(t)

    def __call__(self, t: jax.Array) -> jax.Array:
        # Transparent — child's envelope is already applied via self.pulse(t).
        return self._evaluate(t)


class Summed(Pulse):
    """Additive superposition of multiple pulses.

    Each child pulse is evaluated with its own envelope, then the results
    are summed.  The ``Summed`` pulse's own :attr:`envelope` (inherited
    from :class:`Pulse`, default ``RectangularEnvelope``) is applied on
    top — use ``.windowed(HannEnvelope())`` to window the combined signal.

    Parameters
    ----------
    pulses : tuple[Pulse, ...]
        Pulses to sum.  Must be a tuple (not a list) for Equinox
        PyTree compatibility.
    """

    pulses: tuple[Pulse, ...]

    @property
    def duration(self) -> float:
        # Span from self.initial_time to the latest child endpoint.
        ends = [p.initial_time + p.duration for p in self.pulses]
        return max(ends) - self.initial_time

    @property
    def t_end(self) -> float:
        return max(p.t_end for p in self.pulses)

    def _evaluate(self, t: jax.Array) -> jax.Array:
        # Each p(t) includes the child's own envelope.
        return jnp.sum(jnp.array([p(t) for p in self.pulses]))
