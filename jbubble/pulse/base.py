"""Core abstractions for acoustic driving pulses."""

from __future__ import annotations

import abc
import numbers

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .envelope import Envelope, NoEnvelope, SoftRectangularEnvelope

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

    Operator overloads compose pulses. A factor or offset can be a Python
    number or a JAX scalar, including a traced one, so you can
    differentiate with respect to it:

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
        [`SoftRectangularEnvelope()`][jbubble.pulse.envelope.SoftRectangularEnvelope],
        except for [`Summed`][jbubble.pulse.base.Summed], whose default is
        [`NoEnvelope()`][jbubble.pulse.envelope.NoEnvelope].
    """

    initial_time: float = eqx.field(default=0.0, kw_only=True)
    envelope: Envelope = eqx.field(
        default_factory=SoftRectangularEnvelope, kw_only=True
    )

    # NumPy defers to the reflected operators below, so `array * pulse`
    # builds a `Scaled` pulse rather than an object array of pulses.
    __array_ufunc__ = None

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

    @property
    def window_edges(self) -> jax.Array:
        """Start and end times [s] of every active window in this pulse.

        A leaf pulse returns `[t_start, t_stop]`.
        [`Scaled`][jbubble.pulse.base.Scaled] and
        [`Offset`][jbubble.pulse.base.Offset] return the edges of the pulse
        they wrap, and [`Summed`][jbubble.pulse.base.Summed] returns its own
        edges with those of every child, in ascending order.

        [`solve_eom`][jbubble.solver.solve_eom] makes an adaptive step-size
        controller step to each of these times, so it can't step over a
        pulse that starts late. If you write a pulse that wraps other
        pulses, override this property to include their edges.
        """
        return jnp.stack([jnp.asarray(self.t_start), jnp.asarray(self.t_stop)])

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the pressure [Pa] at time `t`, with the envelope applied."""
        tau = t - self.t_start
        return self._evaluate(t) * self.envelope(tau, self.duration)

    def __add__(self, other: Pulse | ArrayLike) -> Pulse:
        """Add another pulse or a constant offset: `pulse_a + pulse_b` or `pulse + 1.0`.

        Adding two pulses gives one flat [`Summed`][jbubble.pulse.base.Summed].
        A constant offset in either operand moves outside the sum, so
        `(pulse_a + c) + pulse_b` gives
        `Offset(pulse=pulse_a + pulse_b, offset=c)`.
        """
        if isinstance(other, Pulse):
            return _add_pulses(self, other)
        offset = _operand(other)
        if offset is None:
            return NotImplemented
        return Offset(pulse=self, offset=offset)

    def __radd__(self, other: Pulse | ArrayLike) -> Pulse:
        """Right addition, `other + self`. If `other` is a `Pulse`, delegate to its `__add__`."""
        if isinstance(other, Pulse):
            return other.__add__(self)
        offset = _operand(other)
        if offset is None:
            return NotImplemented
        return Offset(pulse=self, offset=offset)

    def __mul__(self, factor: ArrayLike) -> Scaled:
        """Scale the pulse by a factor: `pulse * factor`."""
        value = _operand(factor)
        if value is None:
            return NotImplemented
        return Scaled(pulse=self, factor=value)

    def __rmul__(self, factor: ArrayLike) -> Scaled:
        """Right multiplication: `factor * pulse`."""
        return self.__mul__(factor)

    def __neg__(self) -> Scaled:
        """Negate the pulse (flip its polarity): `-pulse`."""
        return Scaled(pulse=self, factor=-1.0)

    def __pos__(self) -> Pulse:
        """Unary plus (identity): `+pulse`."""
        return self

    def __sub__(self, other: Pulse | ArrayLike) -> Pulse:
        """Subtract another pulse or a constant offset: `pulse_a - pulse_b` or `pulse - 1.0`."""
        if isinstance(other, Pulse):
            return self + (-other)
        offset = _operand(other)
        if offset is None:
            return NotImplemented
        return Offset(pulse=self, offset=-offset)

    def __rsub__(self, other: Pulse | ArrayLike) -> Pulse:
        """Right subtraction: `other - self`."""
        if isinstance(other, Pulse):
            return other + (-self)
        offset = _operand(other)
        if offset is None:
            return NotImplemented
        # other - self = (-self) + other
        return Offset(pulse=-self, offset=offset)

    def __truediv__(self, factor: ArrayLike) -> Scaled:
        """Divide the pulse by a factor: `pulse / 2.0`."""
        value = _operand(factor)
        if value is None:
            return NotImplemented
        return Scaled(pulse=self, factor=1.0 / value)

    def __iadd__(self, other: Pulse | ArrayLike) -> Pulse:
        """In-place addition: `pulse += other` or `pulse += 1.0`."""
        return self.__add__(other)

    def __isub__(self, other: Pulse | ArrayLike) -> Pulse:
        """In-place subtraction: `pulse -= other` or `pulse -= 1.0`."""
        return self.__sub__(other)

    def __imul__(self, factor: ArrayLike) -> Scaled:
        """In-place multiplication: `pulse *= factor`."""
        return self.__mul__(factor)

    def __itruediv__(self, factor: ArrayLike) -> Scaled:
        """In-place division: `pulse /= factor`."""
        return self.__truediv__(factor)

    def windowed(self, envelope: Envelope) -> Pulse:
        """Return a copy of this pulse with `envelope` as its window.

        What the new envelope applies to depends on the kind of pulse:

        - A leaf pulse, such as
          [`ToneBurst`][jbubble.pulse.tone_burst.ToneBurst], replaces its
          own envelope.
        - [`Summed`][jbubble.pulse.base.Summed] replaces its own envelope,
          which multiplies the sum on top of each child's envelope. A sum
          has no window of its own until you call `windowed` on it.
        - [`Scaled`][jbubble.pulse.base.Scaled] and
          [`Offset`][jbubble.pulse.base.Offset] have no envelope of their
          own, so they pass `envelope` to the wrapped pulse. The constant of
          an `Offset` stays outside the window:
          `(pulse + c).windowed(env)` equals `pulse.windowed(env) + c`.
          Addition keeps constants outside sums, so
          `((pulse_a + c) + pulse_b).windowed(env)` also equals
          `(pulse_a + pulse_b).windowed(env) + c`.

        Parameters
        ----------
        envelope : Envelope
            The new window.

        Returns
        -------
        Pulse
            A pulse of the same type as this one.
        """
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
    and it takes its active window (`t_start`, `t_stop`, `duration`,
    `t_end`, and `window_edges`) from the child. It has no start time or
    envelope of its own, so `windowed` passes the new envelope down to the
    child pulse.

    Parameters
    ----------
    pulse : Pulse
        The pulse to scale.
    factor : float or jax.Array
        Multiplicative factor. A JAX scalar, including a traced one, works.

    Raises
    ------
    ValueError
        If you set `initial_time` or `envelope` to anything other than the
        default, because `Scaled` would ignore it. Traced values aren't
        checked.
    """

    pulse: Pulse
    factor: float | jax.Array

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

    @property
    def window_edges(self) -> jax.Array:
        """Window edges of the wrapped pulse [s]."""
        return self.pulse.window_edges

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
    Adding pulses keeps the constant outside the sum, so a window that you
    apply to the sum never gates it: `(pulse_a + c) + pulse_b` gives
    `Offset(pulse=pulse_a + pulse_b, offset=c)`, and `k * (pulse_a + c)`
    contributes `k * c`.

    `Offset` is transparent: it delegates to the child pulse's `__call__`,
    which already applies the child's envelope, and it takes its active
    window (`t_start`, `t_stop`, `duration`, `t_end`, and `window_edges`)
    from the child.
    It has no start time or envelope of its own, so `windowed` passes the
    new envelope down to the child pulse and leaves the constant offset
    unwindowed.

    Parameters
    ----------
    pulse : Pulse
        The pulse to offset.
    offset : float or jax.Array
        Additive constant pressure [Pa]. A JAX scalar, including a traced
        one, works.

    Raises
    ------
    ValueError
        If you set `initial_time` or `envelope` to anything other than the
        default, because `Offset` would ignore it. Traced values aren't
        checked.
    """

    pulse: Pulse
    offset: float | jax.Array

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

    @property
    def window_edges(self) -> jax.Array:
        """Window edges of the wrapped pulse [s]."""
        return self.pulse.window_edges

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
    p(t) = w(t - t_\text{start}, T) \sum_i p_i(t)
    $$

    `Summed` evaluates each child pulse $p_i$ with its own envelope, sums
    the results, and multiplies the sum by its own `envelope` $w$. That
    envelope defaults to [`NoEnvelope`][jbubble.pulse.envelope.NoEnvelope],
    which is 1 at all times, so by default $p(t) = \sum_i p_i(t)$ exactly
    and every child keeps its full signal, however late it starts. To
    window the combined signal, call `.windowed(envelope)`, for example
    `(pulse_a + pulse_b).windowed(HannEnvelope())`.

    The active window runs from the earliest child
    [`t_start`][jbubble.pulse.base.Pulse.t_start] to the latest child
    [`t_stop`][jbubble.pulse.base.Pulse.t_stop], so a window that you apply
    covers the children wherever they start:

    $$
    t_\text{start} = \max\bigl(t_0,\ \min_i t_{\text{start},i}\bigr),
    \qquad
    T = \max_i t_{\text{stop},i} - t_\text{start},
    $$

    where $t_0$ is the sum's own `initial_time`. The window never starts
    before $t_0$, so to start it later than the earliest child, set
    `initial_time`.

    `pulse_a + pulse_b` flattens nested sums into one `Summed`, except a
    windowed sum, whose envelope isn't `NoEnvelope`. That sum stays a
    single child, so it keeps its window. The rule depends only on the
    envelope's type, so a sum flattens the same way under `jax.jit` and
    outside it. Flattening keeps every child but drops a nested sum's own
    `initial_time`, which matters only for a window that you apply later.

    Parameters
    ----------
    pulses : tuple[Pulse, ...]
        Pulses to sum. Must be a tuple, not a list, for Equinox PyTree
        compatibility.
    envelope : Envelope
        Window applied to the sum. Keyword-only. Default:
        [`NoEnvelope()`][jbubble.pulse.envelope.NoEnvelope].
    """

    pulses: tuple[Pulse, ...]
    envelope: Envelope = eqx.field(default_factory=NoEnvelope, kw_only=True)

    @property
    def t_start(self) -> float | jax.Array:
        """Start of the active window [s].

        Equals the earliest child `t_start`, but no earlier than
        `initial_time`.
        """
        starts = jnp.stack([jnp.asarray(p.t_start) for p in self.pulses])
        return jnp.maximum(jnp.asarray(self.initial_time), jnp.min(starts))

    @property
    def duration(self) -> float | jax.Array:
        """Time [s] from `t_start` to the latest child `t_stop`.

        `Scaled` and `Offset` report their child's `t_stop`, so a delayed
        child that is wrapped in them still counts.
        """
        # jnp.max, rather than Python max or float, keeps this valid under
        # JAX tracing.
        stops = jnp.stack([jnp.asarray(p.t_stop) for p in self.pulses])
        return jnp.max(stops) - jnp.asarray(self.t_start)

    @property
    def t_end(self) -> float | jax.Array:
        """Suggested simulation end time [s]: the latest child `t_end`."""
        return jnp.max(jnp.stack([jnp.asarray(p.t_end) for p in self.pulses]))

    @property
    def window_edges(self) -> jax.Array:
        """Window edges [s] of the sum and of every child, in ascending order."""
        edges = [super().window_edges, *(p.window_edges for p in self.pulses)]
        return jnp.sort(jnp.concatenate(edges))

    def _evaluate(self, t: jax.Array) -> jax.Array:
        # Each p(t) includes the child's own envelope.
        return jnp.sum(jnp.array([p(t) for p in self.pulses]))


def _operand(value: object) -> float | jax.Array | None:
    """Return `value` as a factor or offset, or `None` if it isn't one.

    Python and NumPy numbers become Python floats, and 0-d NumPy arrays
    become JAX arrays. 0-d JAX arrays, including traced values, pass
    through unchanged, so you can differentiate or `vmap` over a factor or
    offset.

    Raises
    ------
    ValueError
        If `value` is an array with one or more dimensions.
    """
    if isinstance(value, Pulse):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, (jax.Array, np.ndarray)):
        if value.ndim != 0:
            raise ValueError(
                "Pulse arithmetic takes a scalar factor or offset, but got an "
                f"array of shape {value.shape}. To build one pulse per value, "
                "map over the values with jax.vmap."
            )
        return jnp.asarray(value)
    return None


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


def _summands(pulse: Pulse) -> tuple[Pulse, ...]:
    """Return the pulses that `pulse` contributes to a flattened sum.

    A windowed `Summed`, whose envelope isn't `NoEnvelope`, stays whole, so
    adding to it keeps its window. The test reads only the PyTree
    structure, never a leaf value, so tracing can't change it.
    """
    if type(pulse) is Summed and type(pulse.envelope) is NoEnvelope:
        return pulse.pulses
    return (pulse,)


def _split_offset(pulse: Pulse) -> tuple[Pulse, float | jax.Array | None]:
    """Split `pulse` into a pulse without an outer constant offset, and that constant.

    Looks through `Offset` and `Scaled`, so `k * (p + c)` splits into
    `k * p` and `k * c`. The constant is `None` when there is no offset to
    lift.
    """
    if type(pulse) is Offset:
        inner, constant = _split_offset(pulse.pulse)
        if constant is None:
            return inner, pulse.offset
        return inner, constant + pulse.offset
    if type(pulse) is Scaled:
        inner, constant = _split_offset(pulse.pulse)
        if constant is None:
            return pulse, None
        scaled = eqx.tree_at(lambda s: s.pulse, pulse, inner)
        return scaled, pulse.factor * constant
    return pulse, None


def _add_pulses(a: Pulse, b: Pulse) -> Pulse:
    """Return `a + b` as one flat `Summed`, with constant offsets lifted out.

    A constant inside a sum would be gated by a window applied to the sum
    later, so it goes on an `Offset` around the sum instead.
    """
    a, offset_a = _split_offset(a)
    b, offset_b = _split_offset(b)
    total = Summed(pulses=_summands(a) + _summands(b))
    if offset_a is None:
        return total if offset_b is None else Offset(pulse=total, offset=offset_b)
    if offset_b is None:
        return Offset(pulse=total, offset=offset_a)
    return Offset(pulse=total, offset=offset_a + offset_b)


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
