"""Core abstractions for acoustic driving pulses."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .envelope import Envelope, SoftRectangularEnvelope


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
    envelope: Envelope = eqx.field(
        default_factory=SoftRectangularEnvelope, kw_only=True
    )

    @abc.abstractmethod
    def _evaluate(self, t: jax.Array) -> jax.Array:
        """Raw signal value at time *t*, before envelope application."""
        ...

    @property
    @abc.abstractmethod
    def duration(self) -> float | jax.Array:
        """Active pulse duration [s] (excluding any leading silence)."""
        ...

    @property
    def t_end(self) -> float | jax.Array:
        """Suggested simulation end time [s].

        Default: ``initial_time + 2 × duration``.
        """
        return self.initial_time + 2.0 * self.duration

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate pressure at time *t* [Pa], with envelope applied."""
        tau = t - self.initial_time
        return self._evaluate(t) * self.envelope(tau, self.duration)

    def __add__(self, other: Pulse | float) -> Pulse:
        """Add another pulse or constant offset: pulse_a + pulse_b or pulse + 1.0"""
        if isinstance(other, (int, float)):
            return Offset(pulse=self, offset=float(other))
        left = self.pulses if isinstance(self, Summed) else (self,)
        right = other.pulses if isinstance(other, Summed) else (other,)
        return Summed(pulses=left + right)

    def __radd__(self, other: Pulse | float) -> Pulse:
        """Right addition: other + self.  If *other* is a Pulse, delegate to its __add__."""
        if isinstance(other, (int, float)):
            return Offset(pulse=self, offset=float(other))
        if isinstance(other, Pulse):
            return other.__add__(self)
        return NotImplemented

    def __mul__(self, factor: float) -> Scaled:
        """Scale the pulse by a factor: pulse * factor"""
        return Scaled(pulse=self, factor=float(factor))

    def __rmul__(self, factor: float) -> Scaled:
        """Right multiplication: factor * pulse"""
        return Scaled(pulse=self, factor=float(factor))

    def __neg__(self) -> Scaled:
        """Negate the pulse (flip polarity): -pulse"""
        return Scaled(pulse=self, factor=-1.0)

    def __pos__(self) -> Pulse:
        """Unary plus (identity): +pulse"""
        return self

    def __sub__(self, other: Pulse | float) -> Pulse:
        """Subtract another pulse or constant offset: pulse_a - pulse_b or pulse - 1.0"""
        if isinstance(other, (int, float)):
            return Offset(pulse=self, offset=-float(other))
        return self + (-other)

    def __rsub__(self, other: Pulse | float) -> Pulse:
        """Right subtraction: other - self"""
        if isinstance(other, (int, float)):
            # other - self = (-self) + other
            return Offset(pulse=(-self), offset=float(other))
        if isinstance(other, Pulse):
            return other + (-self)
        return NotImplemented

    def __truediv__(self, factor: float) -> Scaled:
        """Divide pulse by a factor: pulse / 2.0"""
        return Scaled(pulse=self, factor=1.0 / float(factor))

    def __iadd__(self, other: Pulse | float):
        """In-place addition: pulse += other or pulse += 1.0"""
        return self + other

    def __isub__(self, other: Pulse | float):
        """In-place subtraction: pulse -= other or pulse -= 1.0"""
        return self - other

    def __imul__(self, factor: float):
        """In-place multiplication: pulse *= factor"""
        return Scaled(pulse=self, factor=float(factor))

    def __itruediv__(self, factor: float):
        """In-place division: pulse /= factor"""
        return Scaled(pulse=self, factor=1.0 / float(factor))

    def windowed(self, envelope: Envelope) -> Pulse:
        """Return a copy of this pulse with *envelope* replacing the current one."""
        return eqx.tree_at(
            lambda p: p.envelope,
            self,
            envelope,
            is_leaf=lambda x: isinstance(x, Envelope),
        )


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
    def duration(self) -> float | jax.Array:
        return self.pulse.duration

    @property
    def t_end(self) -> float | jax.Array:
        return self.pulse.t_end

    def _evaluate(self, t: jax.Array) -> jax.Array:
        return self.factor * self.pulse(t)

    def __call__(self, t: jax.Array) -> jax.Array:
        # Transparent — child's envelope is already applied via self.pulse(t).
        return self._evaluate(t)


class Offset(Pulse):
    """Constant-offset version of another pulse.

    ``Offset`` is transparent: it delegates entirely to the child pulse's
    ``__call__`` (which already applies the child's envelope) and simply
    adds a constant offset. No additional envelope is applied.

    Parameters
    ----------
    pulse : Pulse
        The pulse to offset.
    offset : float
        Additive constant offset.
    """

    pulse: Pulse
    offset: float

    @property
    def duration(self) -> float | jax.Array:
        return self.pulse.duration

    @property
    def t_end(self) -> float | jax.Array:
        return self.pulse.t_end

    def _evaluate(self, t: jax.Array) -> jax.Array:
        return self.pulse(t) + self.offset

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
        # float() is intentional: Summed uses Python's max(), which requires
        # concrete values — JAX-traced durations are not supported here.
        ends = [float(p.initial_time + p.duration) for p in self.pulses]
        return max(ends) - float(self.initial_time)

    @property
    def t_end(self) -> float:
        return max(float(p.t_end) for p in self.pulses)

    def _evaluate(self, t: jax.Array) -> jax.Array:
        # Each p(t) includes the child's own envelope.
        return jnp.sum(jnp.array([p(t) for p in self.pulses]))
