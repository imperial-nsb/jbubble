"""Core abstractions for acoustic driving pulses."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .envelope import Envelope, SoftRectangularEnvelope

__all__ = ["Pulse", "Scaled", "Offset", "Summed"]


class Pulse(eqx.Module, abc.ABC):
    r"""Abstract acoustic driving pulse.

    Every `Pulse` is callable: `pulse(t)` returns the instantaneous
    pressure [Pa] at time `t`. Implementations must be JAX-differentiable,
    so that equations of motion such as Keller-Miksis can compute
    `jax.grad(pulse)(t)`.

    Subclasses implement `_evaluate`, the raw signal $s(t)$ before the
    envelope. The base `__call__` applies the `envelope` $w$
    automatically:

    $$
    p(t) = s(t)\, w(t - t_0, T)
    $$

    where $t_0$ is `initial_time` and $T$ is
    [`duration`][jbubble.pulse.base.Pulse.duration].

    Operator overloads compose pulses:

    ```python
    combined = pulse_a + pulse_b  # Summed
    scaled = 0.5 * pulse_a  # Scaled
    shifted = pulse_a + 1.0  # Offset
    windowed = combined.windowed(HannEnvelope())  # swaps the envelope
    ```

    Parameters
    ----------
    initial_time : float
        Time at which the pulse starts [s]. Keyword-only. Default: `0.0`.
    envelope : Envelope
        Window applied to the raw signal. Keyword-only. Default:
        [`SoftRectangularEnvelope()`][jbubble.pulse.envelope.SoftRectangularEnvelope].
    """

    initial_time: float = eqx.field(default=0.0, kw_only=True)
    envelope: Envelope = eqx.field(
        default_factory=SoftRectangularEnvelope, kw_only=True
    )

    @abc.abstractmethod
    def _evaluate(self, t: jax.Array) -> jax.Array:
        """Return the raw signal value at time `t`, before the envelope."""
        ...

    @property
    @abc.abstractmethod
    def duration(self) -> float | jax.Array:
        """Active pulse duration [s], excluding any leading silence."""
        ...

    @property
    def t_end(self) -> float | jax.Array:
        """Suggested simulation end time [s].

        The base implementation returns `initial_time + 2 * duration`.
        """
        return self.initial_time + 2.0 * self.duration

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the pressure [Pa] at time `t`, with the envelope applied."""
        tau = t - self.initial_time
        return self._evaluate(t) * self.envelope(tau, self.duration)

    def __add__(self, other: Pulse | float) -> Pulse:
        """Add another pulse or a constant offset: `pulse_a + pulse_b` or `pulse + 1.0`."""
        if isinstance(other, (int, float)):
            return Offset(pulse=self, offset=float(other))
        left = self.pulses if isinstance(self, Summed) else (self,)
        right = other.pulses if isinstance(other, Summed) else (other,)
        return Summed(pulses=left + right)

    def __radd__(self, other: Pulse | float) -> Pulse:
        """Right addition, `other + self`. If `other` is a `Pulse`, delegate to its `__add__`."""
        if isinstance(other, (int, float)):
            return Offset(pulse=self, offset=float(other))
        if isinstance(other, Pulse):
            return other.__add__(self)
        return NotImplemented

    def __mul__(self, factor: float) -> Scaled:
        """Scale the pulse by a factor: `pulse * factor`."""
        return Scaled(pulse=self, factor=float(factor))

    def __rmul__(self, factor: float) -> Scaled:
        """Right multiplication: `factor * pulse`."""
        return Scaled(pulse=self, factor=float(factor))

    def __neg__(self) -> Scaled:
        """Negate the pulse (flip its polarity): `-pulse`."""
        return Scaled(pulse=self, factor=-1.0)

    def __pos__(self) -> Pulse:
        """Unary plus (identity): `+pulse`."""
        return self

    def __sub__(self, other: Pulse | float) -> Pulse:
        """Subtract another pulse or a constant offset: `pulse_a - pulse_b` or `pulse - 1.0`."""
        if isinstance(other, (int, float)):
            return Offset(pulse=self, offset=-float(other))
        return self + (-other)

    def __rsub__(self, other: Pulse | float) -> Pulse:
        """Right subtraction: `other - self`."""
        if isinstance(other, (int, float)):
            # other - self = (-self) + other
            return Offset(pulse=(-self), offset=float(other))
        if isinstance(other, Pulse):
            return other + (-self)
        return NotImplemented

    def __truediv__(self, factor: float) -> Scaled:
        """Divide the pulse by a factor: `pulse / 2.0`."""
        return Scaled(pulse=self, factor=1.0 / float(factor))

    def __iadd__(self, other: Pulse | float):
        """In-place addition: `pulse += other` or `pulse += 1.0`."""
        return self + other

    def __isub__(self, other: Pulse | float):
        """In-place subtraction: `pulse -= other` or `pulse -= 1.0`."""
        return self - other

    def __imul__(self, factor: float):
        """In-place multiplication: `pulse *= factor`."""
        return Scaled(pulse=self, factor=float(factor))

    def __itruediv__(self, factor: float):
        """In-place division: `pulse /= factor`."""
        return Scaled(pulse=self, factor=1.0 / float(factor))

    def windowed(self, envelope: Envelope) -> Pulse:
        """Return a copy of this pulse with `envelope` replacing the current one."""
        return eqx.tree_at(
            lambda p: p.envelope,
            self,
            envelope,
            is_leaf=lambda x: isinstance(x, Envelope),
        )


class Scaled(Pulse):
    """Amplitude-scaled version of another pulse.

    `Scaled` is transparent: it delegates entirely to the child pulse's
    `__call__`, which already applies the child's envelope, and multiplies
    the result by `factor`. It applies no additional envelope, so
    `windowed` passes the new envelope down to the child pulse.

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

    def windowed(self, envelope: Envelope) -> Pulse:
        # Scaled ignores its own envelope, so setting it here would be a silent
        # no-op.  Push the window down to the child pulse, which actually
        # applies envelopes, so ((a + b) * k).windowed(env) windows the sum.
        return eqx.tree_at(lambda p: p.pulse, self, self.pulse.windowed(envelope))


class Offset(Pulse):
    """Constant-offset version of another pulse.

    `Offset` is transparent: it delegates entirely to the child pulse's
    `__call__`, which already applies the child's envelope, and adds a
    constant offset. It applies no additional envelope, so `windowed`
    passes the new envelope down to the child pulse and leaves the constant
    offset unwindowed.

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

    def windowed(self, envelope: Envelope) -> Pulse:
        # Offset ignores its own envelope; push the window down to the child.
        # The constant offset itself stays unwindowed.
        return eqx.tree_at(lambda p: p.pulse, self, self.pulse.windowed(envelope))


class Summed(Pulse):
    """Additive superposition of multiple pulses.

    `Summed` evaluates each child pulse with its own envelope, then sums
    the results. It applies its own `envelope` on top. `Summed` inherits
    that field from [`Pulse`][jbubble.pulse.base.Pulse], and it defaults to
    [`SoftRectangularEnvelope`][jbubble.pulse.envelope.SoftRectangularEnvelope].
    To window the combined signal, use `.windowed(HannEnvelope())`.

    Parameters
    ----------
    pulses : tuple[Pulse, ...]
        Pulses to sum. Must be a tuple, not a list, for Equinox PyTree
        compatibility.
    """

    pulses: tuple[Pulse, ...]

    @property
    def duration(self) -> float | jax.Array:
        # Span from self.initial_time to the latest child endpoint.  Uses
        # jnp.max over stacked child endpoints (not Python max/float) so it
        # stays valid under JAX tracing, so a Summed pulse can be simulated.
        ends = jnp.stack(
            [jnp.asarray(p.initial_time + p.duration) for p in self.pulses]
        )
        return jnp.max(ends) - jnp.asarray(self.initial_time)

    @property
    def t_end(self) -> float | jax.Array:
        return jnp.max(jnp.stack([jnp.asarray(p.t_end) for p in self.pulses]))

    def _evaluate(self, t: jax.Array) -> jax.Array:
        # Each p(t) includes the child's own envelope.
        return jnp.sum(jnp.array([p(t) for p in self.pulses]))
