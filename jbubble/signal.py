"""Flexible driving-signal abstractions for bubble excitation.

The :class:`DrivingSignal` base class defines the interface that any driving
pressure must satisfy: it is callable as ``signal(t) -> pressure`` and can be
non-dimensionalised via :meth:`get_scaled`.  Concrete implementations include:

* :class:`~jbubble.pulse.Pulse` – analytic, frequency-based waveforms built
  from a carrier frequency, amplitude, and envelope/shape.
* :class:`Waveform` – arbitrary user-supplied time-domain samples with
  differentiable linear interpolation.

Future directions (accommodated by this base class) include spectrogram-based
signals, where a time–frequency matrix controls which frequency bins are active
over time, and composite multi-segment signals.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .units import Units


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DrivingSignal(eqx.Module):
    """Abstract base for any time-dependent driving pressure.

    Every concrete subclass must implement:

    * ``__call__(t)`` – return the instantaneous driving pressure at time *t*.
    * ``get_scaled(units)`` – return a copy expressed in non-dimensional units.
    * ``duration`` – the nominal active duration of the signal.
    """

    @abc.abstractmethod
    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the driving pressure at time *t*."""

    @abc.abstractmethod
    def get_scaled(self, units: Units) -> "DrivingSignal":
        """Return a non-dimensionalised copy of this signal."""

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """Nominal active duration of the signal (seconds, or non-dim)."""

    @property
    def start_time(self) -> float:
        """Time at which the signal nominally begins (default ``0.0``)."""
        return 0.0


# ---------------------------------------------------------------------------
# Arbitrary time-domain waveform
# ---------------------------------------------------------------------------


class Waveform(DrivingSignal):
    """Arbitrary time-domain driving pressure from uniformly-spaced samples.

    The waveform is reconstructed at arbitrary query times via differentiable
    linear interpolation.  Outside the sample window
    ``[t0, t0 + (N-1) * dt]`` the returned pressure is zero.

    Parameters
    ----------
    samples : array-like
        1-D array of pressure values (Pa, or non-dimensional after scaling).
        Converted to a :class:`jax.Array` on construction.
    dt : float
        Uniform time step between consecutive samples.
    t0 : float, optional
        Start time of the waveform (default ``0.0``).
    """

    samples: jax.Array
    dt: float
    t0: float

    def __init__(self, samples, dt: float, t0: float = 0.0):
        self.samples = jnp.asarray(samples)
        self.dt = float(dt)
        self.t0 = float(t0)

    # -- DrivingSignal interface -----------------------------------------------

    def __call__(self, t: jax.Array) -> jax.Array:
        tau = t - self.t0
        n = self.samples.shape[0]
        t_end = (n - 1) * self.dt

        # Fractional sample index
        idx_f = tau / self.dt
        idx_lo = jnp.clip(jnp.floor(idx_f).astype(jnp.int32), 0, n - 2)
        frac = jnp.clip(idx_f - idx_lo, 0.0, 1.0)

        # Differentiable linear interpolation
        val = self.samples[idx_lo] * (1.0 - frac) + self.samples[idx_lo + 1] * frac

        # Zero outside the waveform window
        in_range = (tau >= 0.0) & (tau <= t_end)
        return jnp.where(in_range, val, 0.0)

    def get_scaled(self, units: Units) -> "Waveform":
        return Waveform(
            samples=self.samples / units.P_scale,
            dt=self.dt / units.T_scale,
            t0=self.t0 / units.T_scale,
        )

    @property
    def duration(self) -> float:
        return (self.samples.shape[0] - 1) * self.dt

    @property
    def start_time(self) -> float:
        return self.t0

    # -- convenience constructors ----------------------------------------------

    @staticmethod
    def from_function(
        fn,
        t0: float,
        t1: float,
        n_samples: int,
    ) -> "Waveform":
        """Sample a callable ``fn(t) -> pressure`` on a uniform grid.

        Useful for converting an analytic expression into a tabulated waveform,
        e.g. for testing or for feeding to tools that expect a :class:`Waveform`.
        """
        dt = (t1 - t0) / (n_samples - 1)
        ts = jnp.linspace(t0, t1, n_samples)
        samples = jax.vmap(fn)(ts)
        return Waveform(samples=samples, dt=dt, t0=t0)

    @staticmethod
    def from_pulse(
        pulse: "Pulse",
        n_samples: int = 4096,
        pad_factor: float = 1.5,
    ) -> "Waveform":
        """Convert an existing :class:`~jbubble.pulse.Pulse` to a tabulated waveform.

        Parameters
        ----------
        pulse : Pulse
            The analytic pulse to sample.
        n_samples : int
            Number of uniformly-spaced time samples.
        pad_factor : float
            Fraction of pulse duration to add after the pulse ends (captures
            the zero tail).
        """
        t0 = pulse.initial_time
        t1 = t0 + pulse.cycle_num / pulse.freq * pad_factor
        dt = (t1 - t0) / (n_samples - 1)
        ts = jnp.linspace(t0, t1, n_samples)
        samples = jax.vmap(pulse)(ts)
        return Waveform(samples=samples, dt=dt, t0=t0)
