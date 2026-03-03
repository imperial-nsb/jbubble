"""Driving pulse parameterisations."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .shapes import PulseShape
from .units import Units


class Pulse(eqx.Module):
    """Driving acoustic pulse — envelope × waveform shape × pressure amplitude.

    A ``Pulse`` is callable: ``pulse(t)`` returns the instantaneous pressure
    [same units as ``pressure``] at time ``t``.

    Parameters
    ----------
    freq : float
        Carrier frequency [Hz].
    pressure : float
        Peak pressure amplitude [Pa].
    shape : PulseShape
        Waveform shape (``Sine``, ``Sawtooth``, …).  The shape is normalised
        so that its peak absolute value is 1.
    phase : float
        Carrier phase offset [rad]. Default: 0.
    initial_time : float
        Time at which the pulse starts [s]. The pulse is zero for
        ``t < initial_time``. Default: 0.
    cycle_num : float
        Number of carrier cycles in the burst. Default: 4.
    apply_hann : bool
        If ``True``, multiply the burst by a Hann (raised-cosine) window to
        give a smooth on/off transition and suppress spectral leakage.
        Default: ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jbubble import Pulse
    >>> from jbubble.shapes import Sine
    >>> pulse = Pulse(freq=1e6, pressure=100e3, shape=Sine(), cycle_num=5)
    >>> float(pulse(jnp.array(0.0)))  # before initial_time
    0.0
    """

    freq: float
    pressure: float
    shape: PulseShape
    phase: float = 0.0
    initial_time: float = 0.0
    cycle_num: float = 4.0
    apply_hann: bool = eqx.field(default=False, static=True)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the pulse pressure at time *t*.

        Parameters
        ----------
        t : jax.Array
            Time [same units as ``initial_time``].

        Returns
        -------
        jax.Array
            Instantaneous pressure [Pa or dimensionless if scaled].
        """
        pulse_span = self.cycle_num / self.freq
        tau = t - self.initial_time
        in_pulse = (tau >= 0) & (tau <= pulse_span)

        hann = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau / pulse_span))
        window_val = jnp.where(self.apply_hann, hann, 1.0)
        window = jnp.where(in_pulse, window_val, 0.0)

        val = self.shape(t, self.freq, self.phase, self.initial_time)
        return val * self.pressure * window

    def get_scaled(self, units: Units) -> "Pulse":
        """Return a dimensionless copy of this pulse scaled by *units*.

        Parameters
        ----------
        units : Units
            Non-dimensionalisation factors.

        Returns
        -------
        Pulse
            Copy with ``freq``, ``pressure``, and ``initial_time`` divided by
            their respective scale factors.
        """
        return Pulse(
            freq=self.freq / units.freq_scale,
            pressure=self.pressure / units.P_scale,
            shape=self.shape,
            phase=self.phase,
            initial_time=self.initial_time / units.T_scale,
            cycle_num=self.cycle_num,
            apply_hann=self.apply_hann,
        )
