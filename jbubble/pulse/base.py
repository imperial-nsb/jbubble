"""Core abstractions for acoustic driving pulses."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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
    p(t) = s(t)\, w(t - t_\text{start}, T)
    $$

    where $T$ is [`duration`][jbubble.pulse.base.Pulse.duration] and
    $t_\text{start}$ is [`t_start`][jbubble.pulse.base.Pulse.t_start],
    which equals `initial_time` for most pulses. The active window runs
    from `t_start` to [`t_stop`][jbubble.pulse.base.Pulse.t_stop].

    Operator overloads compose pulses:

    ```python
    combined = pulse_a + pulse_b  # Summed
    scaled = 0.5 * pulse_a  # Scaled
    shifted = pulse_a + 1.0  # Offset: a constant pressure, not a delay
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
    def t_start(self) -> float | jax.Array:
        """Absolute time [s] at which the active window starts.

        The base implementation returns `initial_time`.
        """
        return self.initial_time

    @property
    def t_stop(self) -> float | jax.Array:
        """Absolute time [s] at which the active window ends.

        Equals `t_start + duration`.
        """
        return self.t_start + self.duration

    @property
    def t_end(self) -> float | jax.Array:
        """Suggested simulation end time [s].

        The base implementation returns `t_start + 2 * duration`.
        """
        return self.t_start + 2.0 * self.duration

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the pressure [Pa] at time `t`, with the envelope applied."""
        tau = t - self.t_start
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
    r"""Amplitude-scaled version of another pulse.

    $$
    p(t) = k\, p_\text{child}(t)
    $$

    where $k$ is `factor`. `Scaled` is transparent: it delegates to the
    child pulse's `__call__`, which already applies the child's envelope,
    and it takes its active window (`t_start`, `t_stop`, `duration`, and
    `t_end`) from the child. It has no start time or envelope of its own,
    so `windowed` passes the new envelope down to the child pulse.

    Parameters
    ----------
    pulse : Pulse
        The pulse to scale.
    factor : float
        Multiplicative factor.

    Raises
    ------
    ValueError
        If you set `initial_time` or `envelope` to anything other than the
        default, because `Scaled` would ignore it. Traced values aren't
        checked.
    """

    pulse: Pulse
    factor: float

    def __check_init__(self) -> None:
        _check_transparent(self)

    @property
    def duration(self) -> float | jax.Array:
        """Duration of the wrapped pulse [s]."""
        return self.pulse.duration

    @property
    def t_start(self) -> float | jax.Array:
        """Start of the wrapped pulse's active window [s]."""
        return self.pulse.t_start

    @property
    def t_stop(self) -> float | jax.Array:
        """End of the wrapped pulse's active window [s]."""
        return self.pulse.t_stop

    @property
    def t_end(self) -> float | jax.Array:
        """Suggested simulation end time of the wrapped pulse [s]."""
        return self.pulse.t_end

    def _evaluate(self, t: jax.Array) -> jax.Array:
        return self.factor * self.pulse(t)

    def __call__(self, t: jax.Array) -> jax.Array:
        # Transparent: self.pulse(t) already applies the child's envelope.
        return self._evaluate(t)

    def windowed(self, envelope: Envelope) -> Pulse:
        # Scaled has no envelope of its own. Push the window down to the
        # child, so ((a + b) * k).windowed(env) windows the sum.
        return eqx.tree_at(lambda p: p.pulse, self, self.pulse.windowed(envelope))


class Offset(Pulse):
    r"""Constant-offset version of another pulse.

    $$
    p(t) = p_\text{child}(t) + c
    $$

    where $c$ is `offset`, a constant pressure added at all times. It
    doesn't delay the pulse; to delay a pulse, set `initial_time` on it.

    `Offset` is transparent: it delegates to the child pulse's `__call__`,
    which already applies the child's envelope, and it takes its active
    window (`t_start`, `t_stop`, `duration`, and `t_end`) from the child.
    It has no start time or envelope of its own, so `windowed` passes the
    new envelope down to the child pulse and leaves the constant offset
    unwindowed.

    Parameters
    ----------
    pulse : Pulse
        The pulse to offset.
    offset : float
        Additive constant pressure [Pa].

    Raises
    ------
    ValueError
        If you set `initial_time` or `envelope` to anything other than the
        default, because `Offset` would ignore it. Traced values aren't
        checked.
    """

    pulse: Pulse
    offset: float

    def __check_init__(self) -> None:
        _check_transparent(self)

    @property
    def duration(self) -> float | jax.Array:
        """Duration of the wrapped pulse [s]."""
        return self.pulse.duration

    @property
    def t_start(self) -> float | jax.Array:
        """Start of the wrapped pulse's active window [s]."""
        return self.pulse.t_start

    @property
    def t_stop(self) -> float | jax.Array:
        """End of the wrapped pulse's active window [s]."""
        return self.pulse.t_stop

    @property
    def t_end(self) -> float | jax.Array:
        """Suggested simulation end time of the wrapped pulse [s]."""
        return self.pulse.t_end

    def _evaluate(self, t: jax.Array) -> jax.Array:
        return self.pulse(t) + self.offset

    def __call__(self, t: jax.Array) -> jax.Array:
        # Transparent: self.pulse(t) already applies the child's envelope.
        return self._evaluate(t)

    def windowed(self, envelope: Envelope) -> Pulse:
        # Offset has no envelope of its own; push the window down to the
        # child. The constant offset itself stays unwindowed.
        return eqx.tree_at(lambda p: p.pulse, self, self.pulse.windowed(envelope))


class Summed(Pulse):
    r"""Additive superposition of multiple pulses.

    $$
    p(t) = w(t - t_0, T) \sum_i p_i(t)
    $$

    `Summed` evaluates each child pulse $p_i$ with its own envelope, sums
    the results, and applies its own `envelope` $w$ on top. `Summed`
    inherits that field from [`Pulse`][jbubble.pulse.base.Pulse], and it
    defaults to
    [`SoftRectangularEnvelope`][jbubble.pulse.envelope.SoftRectangularEnvelope].
    Its active window runs from its own `initial_time` $t_0$ to the latest
    child [`t_stop`][jbubble.pulse.base.Pulse.t_stop], so
    $T = \max_i t_{\text{stop},i} - t_0$. To window the combined signal,
    use `.windowed(HannEnvelope())`.

    Parameters
    ----------
    pulses : tuple[Pulse, ...]
        Pulses to sum. Must be a tuple, not a list, for Equinox PyTree
        compatibility.
    """

    pulses: tuple[Pulse, ...]

    @property
    def duration(self) -> float | jax.Array:
        # Span from t_start to the latest child t_stop. Scaled and Offset
        # report their child's t_stop, so a delayed child that is wrapped
        # in them still counts. jnp.max, rather than Python max or float,
        # keeps this valid under JAX tracing.
        stops = jnp.stack([jnp.asarray(p.t_stop) for p in self.pulses])
        return jnp.max(stops) - jnp.asarray(self.t_start)

    @property
    def t_end(self) -> float | jax.Array:
        return jnp.max(jnp.stack([jnp.asarray(p.t_end) for p in self.pulses]))

    def _evaluate(self, t: jax.Array) -> jax.Array:
        # Each p(t) includes the child's own envelope.
        return jnp.sum(jnp.array([p(t) for p in self.pulses]))


def _concrete(value: object) -> np.ndarray | None:
    """Return `value` as a NumPy array, or `None` if it is a JAX tracer."""
    try:
        return np.asarray(value)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


def _equals(value: object, default: object) -> bool | None:
    """Return whether the PyTree `value` equals `default`.

    Returns `None` when a traced leaf hides the answer.
    """
    if jax.tree.structure(value) != jax.tree.structure(default):
        return False
    unknown = False
    leaves = zip(jax.tree.leaves(value), jax.tree.leaves(default), strict=True)
    for leaf, default_leaf in leaves:
        concrete = _concrete(leaf)
        if concrete is None:
            unknown = True
        elif not np.array_equal(concrete, default_leaf):
            return False
    return None if unknown else True


def _default_initial_time(pulse: Pulse) -> bool | None:
    return _equals(pulse.initial_time, 0.0)


def _default_envelope(pulse: Pulse) -> bool | None:
    return _equals(pulse.envelope, SoftRectangularEnvelope())


def _check_transparent(pulse: Scaled | Offset) -> None:
    """Reject an `initial_time` or `envelope` that `pulse` would ignore."""
    name = type(pulse).__name__
    if _default_initial_time(pulse) is False:
        raise ValueError(
            f"{name} has no start time of its own, so it can't use "
            f"initial_time={pulse.initial_time!r}. Set initial_time on the "
            "wrapped pulse instead."
        )
    if _default_envelope(pulse) is False:
        raise ValueError(
            f"{name} has no envelope of its own, so it can't use "
            f"envelope={pulse.envelope!r}. Call .windowed(envelope) instead, "
            "which passes the envelope to the wrapped pulse."
        )
