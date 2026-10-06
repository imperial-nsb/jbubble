r"""Fit model parameters by differentiating through the ODE solve.

[`fit_parameters`][jbubble.fitting.fit_parameters] minimises a loss computed
from one or more simulations with any optax optimiser. Wrap a value in
[`Parameter`][jbubble.fitting.Parameter] to give it bounds or a scale, or to
hold it fixed. [`unwrap`][jbubble.fitting.unwrap] replaces every `Parameter`
in a pytree by its physical value, for your own training loops.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax.typing import ArrayLike

from .bubble.eom import EquationOfMotion
from .pulse import Pulse
from .simulation import SimulationResult, _simulate
from .solver import SaveSpec, SolverConfig

PyTree = Any  # any JAX-compatible pytree (scalar, array, dict, eqx.Module, ...)

__all__ = ["FitResult", "Parameter", "fit_parameters", "unwrap"]


# ── parameters ───────────────────────────────────────────────────────────────


class Parameter(eqx.Module):
    r"""A fitted quantity, with optional bounds, a scale, or a fixed value.

    The optimiser never updates the physical value $x$ directly. It updates
    an unconstrained coordinate $u$ of order one, and $x$ follows from $u$:

    $$
    x = \begin{cases}
    s\,u & \text{no bounds} \\
    a + s\,e^{u} & \text{lower bound } a \\
    b - s\,e^{u} & \text{upper bound } b \\
    a + (b - a)\,\operatorname{sigmoid}(u) & \text{both bounds}
    \end{cases}
    $$

    where $s$ is `scale`. The default `scale` is $|x_0|$ without bounds and
    the distance from the initial value $x_0$ to the bound with one bound.
    So an optimiser step of 0.05 in $u$ changes $x$ by about 5 %, whatever
    its units, and one learning rate suits `kappa_s` near $10^{-9}$ N s/m and
    `chi` near 0.5 N/m alike. With bounds, $x$ stays strictly inside them.

    Build a `Parameter` outside `jax.jit` and `jax.vmap`: the constructor
    checks the value and bounds on concrete numbers.

    Parameters
    ----------
    value : float or array
        Initial physical value, in SI units. An array is fitted element by
        element.
    lower : float, optional
        Lower bound. `value` must be greater than it. A scalar: one bound
        applies to every element of an array `value`. `-inf`, as in SciPy,
        means no lower bound, the same as `None`.
    upper : float, optional
        Upper bound. `value` must be less than it. A scalar, like `lower`.
        `inf` means no upper bound, the same as `None`.
    scale : float, optional
        Typical magnitude of $x$ (no bounds) or of its distance from the
        bound (one bound). Required when `value` is zero, there are no
        bounds, and the parameter isn't fixed, for example
        `Parameter(0.0, scale=1e-7)` for a trigger delay [s]. Ignored with
        two bounds. `scale` is a static field, so `Parameter`s with
        different scales have different pytree structures, and a compiled
        function compiles again for each one. By default, it comes from
        `value`: to reuse a compiled function for several initial values,
        pass the same `scale` to each.
    fixed : bool
        Whether to hold $x$ at `value`. Use this to switch a parameter off
        without editing `make_model`, or to hold a value inside an Equinox
        module fixed without a warning from
        [`fit_parameters`][jbubble.fitting.fit_parameters]. Default:
        `False`.

    Attributes
    ----------
    raw : jax.Array
        The coordinate $u$ that the optimiser updates. For a fixed
        parameter, the value itself.

    Raises
    ------
    ValueError
        If `value` isn't finite, lies outside the bounds, or is zero
        without bounds, `scale`, or `fixed`, or if a bound is NaN,
        `lower >= upper`, or `scale` isn't positive and finite.

    Examples
    --------
    ```python
    kappa_s = Parameter(1e-9, lower=0.0)  # positive, log-scaled [N s/m]
    chi = Parameter(0.3, lower=0.0, upper=1.5)  # bounded [N/m]
    R0 = Parameter(2e-6, fixed=True)  # held fixed [m]
    ```
    """

    raw: jax.Array
    lower: float | None = eqx.field(static=True)
    upper: float | None = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    fixed: bool = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        lower: float | None = None,
        upper: float | None = None,
        *,
        scale: float | None = None,
        fixed: bool = False,
    ):
        v = jnp.asarray(value, dtype=float)
        if not bool(jnp.all(jnp.isfinite(v))):
            raise ValueError(f"Parameter: value must be finite, got {value!r}.")
        lower = None if lower is None else float(lower)
        upper = None if upper is None else float(upper)
        if (lower is not None and lower != lower) or (
            upper is not None and upper != upper
        ):
            raise ValueError(
                f"Parameter: lower and upper can't be NaN, got lower={lower}, "
                f"upper={upper}. Omit a bound, or pass None, for no bound."
            )
        # An infinite bound on its own side is no bound, as in SciPy.
        self.lower = None if lower == float("-inf") else lower
        self.upper = None if upper == float("inf") else upper
        self.fixed = bool(fixed)
        lo, hi = self.lower, self.upper
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(f"Parameter: lower={lo} must be less than upper={hi}.")
        if lo is not None and not bool(jnp.all(v > lo)):
            raise ValueError(
                f"Parameter: value {value!r} must be greater than lower={lo}."
            )
        if hi is not None and not bool(jnp.all(v < hi)):
            raise ValueError(
                f"Parameter: value {value!r} must be less than upper={hi}."
            )
        if scale is not None and not (0.0 < float(scale) < float("inf")):
            raise ValueError(
                f"Parameter: scale must be positive and finite, got {scale!r}."
            )

        if lo is not None and hi is not None:
            self.scale = hi - lo
        elif scale is not None:
            self.scale = float(scale)
        elif lo is not None:
            self.scale = float(jnp.exp(jnp.mean(jnp.log(v - lo))))
        elif hi is not None:
            self.scale = float(jnp.exp(jnp.mean(jnp.log(hi - v))))
        else:
            magnitude = float(jnp.max(jnp.abs(v)))
            if magnitude == 0.0 and self.fixed:
                magnitude = 1.0  # a fixed parameter never uses its scale
            elif magnitude == 0.0:
                raise ValueError(
                    "Parameter: can't infer a scale for a value of zero. Pass "
                    "scale=<typical magnitude>, for example Parameter(0.0, scale=1e-7)."
                )
            self.scale = magnitude

        if self.fixed:
            self.raw = v
        elif lo is not None and hi is not None:
            z = (v - lo) / (hi - lo)
            self.raw = jnp.log(z) - jnp.log1p(-z)
        elif lo is not None:
            self.raw = jnp.log((v - lo) / self.scale)
        elif hi is not None:
            self.raw = jnp.log((hi - v) / self.scale)
        else:
            self.raw = v / self.scale

    @property
    def value(self) -> jax.Array:
        """Physical value $x$ of the parameter, in SI units."""
        if self.fixed:
            return jax.lax.stop_gradient(self.raw)
        u = self.raw
        if self.lower is not None and self.upper is not None:
            return self.lower + (self.upper - self.lower) * jax.nn.sigmoid(u)
        if self.lower is not None:
            return self.lower + self.scale * jnp.exp(u)
        if self.upper is not None:
            return self.upper - self.scale * jnp.exp(u)
        return self.scale * u


def _is_parameter(x: Any) -> bool:
    return isinstance(x, Parameter)


def _is_module(x: Any) -> bool:
    return isinstance(x, eqx.Module)


def unwrap(tree: PyTree) -> PyTree:
    """Replace every [`Parameter`][jbubble.fitting.Parameter] in `tree` by its value.

    [`fit_parameters`][jbubble.fitting.fit_parameters] calls this for you
    before it calls `make_model`. Call it yourself in a hand-written training
    loop.

    Parameters
    ----------
    tree : PyTree
        Any pytree, such as a dict of `Parameter`s or an Equinox module with
        `Parameter` fields.

    Returns
    -------
    PyTree
        `tree`, with each `Parameter` replaced by its physical value, a
        `jax.Array`.
    """
    return jtu.tree_map(
        lambda x: x.value if _is_parameter(x) else x, tree, is_leaf=_is_parameter
    )


def _path_name(path: tuple) -> str:
    """`{"shell": {"chi": ...}}` -> `"shell.chi"`."""
    parts = []
    for key in path:
        if isinstance(key, jtu.DictKey):
            parts.append(str(key.key))
        elif isinstance(key, jtu.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jtu.SequenceKey):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return ".".join(parts) or "params"


def _canonicalise(params0: PyTree) -> PyTree:
    """Turn every Python float outside an Equinox module into a `Parameter`."""

    def convert(path: tuple, x: Any) -> Any:
        if isinstance(x, float):
            try:
                return Parameter(x)
            except ValueError as error:
                raise ValueError(
                    f"params0 entry {_path_name(path)!r}: {error}"
                ) from None
        return x

    return jtu.tree_map_with_path(convert, params0, is_leaf=_is_module)


def _held_floats(spec: PyTree) -> list[str]:
    """Name the Python floats inside Equinox modules that you set.

    A float field that equals its default, such as a pulse's
    `initial_time=0.0`, is left out: you didn't choose it, so you're
    unlikely to expect it to be fitted. A float inside a dict, list, or
    tuple field has no default to compare with, so it's always named.
    """
    names: list[str] = []

    def visit(path: tuple, node: Any) -> None:
        if not _is_module(node) or _is_parameter(node):
            return
        for field in dataclasses.fields(node):
            if field.metadata.get("static", False):
                continue
            value = getattr(node, field.name, None)
            where = (*path, jtu.GetAttrKey(field.name))
            if isinstance(value, float):
                if not (isinstance(field.default, float) and value == field.default):
                    names.append(_path_name(where))
                continue
            for sub, leaf in jtu.tree_leaves_with_path(value, is_leaf=_is_module):
                if isinstance(leaf, float):
                    names.append(_path_name((*where, *sub)))
                else:
                    visit((*where, *sub), leaf)

    for path, leaf in jtu.tree_leaves_with_path(spec, is_leaf=_is_module):
        visit(path, leaf)
    return names


def _warn_held_floats(spec: PyTree) -> None:
    names = _held_floats(spec)
    if not names:
        return
    shown = ", ".join(names[:5])
    if len(names) > 5:
        shown += f", and {len(names) - 5} more"
    warnings.warn(
        f"fit_parameters: holding fixed {len(names)} Python float(s) inside "
        f"Equinox modules: {shown}. To fit one, wrap it in Parameter(...). To "
        "hold it fixed without this warning, wrap it in Parameter(..., "
        "fixed=True), or declare the field with eqx.field(static=True).",
        UserWarning,
        stacklevel=3,
    )


def _trainable_filter(spec: PyTree) -> PyTree:
    """`True` for the leaves that the optimiser updates."""
    return jtu.tree_map(
        lambda x: (not x.fixed) if _is_parameter(x) else eqx.is_inexact_array(x),
        spec,
        is_leaf=_is_parameter,
    )


def _check_arity(fn: Callable, name: str, args: tuple[str, ...]) -> None:
    """Raise a TypeError that names `fn` if it can't take `args` positionally."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # no signature to check, for example a builtin
        return
    try:
        signature.bind(*args)
    except TypeError:
        hint = (
            "with conditions, it also receives each condition."
            if len(args) == 2
            else "to give it a second argument, pass conditions=[...]."
        )
        raise TypeError(
            f"fit_parameters: {name} must take {len(args)} positional "
            f"argument(s), {name}({', '.join(args)}); {hint}"
        ) from None


# ── conditions and simulation ────────────────────────────────────────────────


def _is_model(x: Any) -> bool:
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and isinstance(x[0], EquationOfMotion)
        and isinstance(x[1], Pulse)
    )


def _stack(conditions: Sequence[PyTree]) -> PyTree | None:
    """Stack conditions along a new leading axis, or return `None` if they differ.

    Conditions stack when they have the same pytree structure, at least one
    leaf, and every leaf is numeric with the same shape in every condition.
    A Python bool is a configuration flag, not data: stacking would turn it
    into a traced array that `make_model` can't use in an `if`, so a bool
    keeps the conditions sequential, where it stays a Python value.
    """
    if len(conditions) < 2:
        return None
    flat = [jtu.tree_flatten(c) for c in conditions]
    treedef = flat[0][1]
    if not flat[0][0] or any(td != treedef for _, td in flat):
        return None
    columns = []
    for column in zip(*(leaves for leaves, _ in flat), strict=True):
        if not all(
            eqx.is_array_like(x) and not isinstance(x, (str, bytes, bool))
            for x in column
        ):
            return None
        arrays = [jnp.asarray(x) for x in column]
        if any(a.shape != arrays[0].shape for a in arrays):
            return None
        columns.append(jnp.stack(arrays))
    return jtu.tree_unflatten(treedef, columns)


def _all_finite(tree: PyTree) -> jax.Array:
    leaves = [jnp.all(jnp.isfinite(x)) for x in jtu.tree_leaves(tree)]
    return jnp.all(jnp.stack(leaves)) if leaves else jnp.asarray(True)


# ── reporting ────────────────────────────────────────────────────────────────


def _format_params(params: PyTree) -> str:
    leaves = [
        (_path_name(path), x)
        for path, x in jtu.tree_leaves_with_path(params)
        if eqx.is_inexact_array(x)
    ]
    if len(leaves) > 6 or any(x.ndim > 0 for _, x in leaves):
        return f"<{sum(x.size for _, x in leaves)} values in {len(leaves)} arrays>"
    return ", ".join(f"{name}={float(x):.4g}" for name, x in leaves)


def _warn_on_first_step(reference: PyTree, updates: PyTree) -> None:
    """Warn once if the first update is badly scaled for some coordinate.

    `reference` holds the typical magnitude of each coordinate: 1 for a
    `Parameter` coordinate, which is scaled by construction, and the value
    itself for a raw array. Adam-type optimisers take first steps of about
    the learning rate in each coordinate's units, so a step larger than the
    reference (`kappa_s = jnp.array(1e-9)` with `adam(1e-2)`) or below a
    millionth of it (`adam(1e-10)`) is almost always a mistake.
    """
    too_big, too_small = [], []
    for (path, ref), u in zip(
        jtu.tree_leaves_with_path(reference), jtu.tree_leaves(updates), strict=True
    ):
        ref_norm = float(jnp.linalg.norm(jnp.ravel(ref)))
        u_norm = float(jnp.linalg.norm(jnp.ravel(u)))
        if ref_norm == 0.0 or u_norm == 0.0:
            continue
        ratio = u_norm / ref_norm
        name = _path_name(path)
        name = "params" if name == "raw" else name.removesuffix(".raw")
        if ratio > 1.0:
            too_big.append(f"{name} (step = {ratio:.1e} x its scale)")
        elif ratio < 1e-6:
            too_small.append(f"{name} (step = {ratio:.1e} x its scale)")
    if too_big or too_small:
        parts = []
        if too_big:
            parts.append("the first step is larger than " + ", ".join(too_big))
        if too_small:
            parts.append("the first step barely changes " + ", ".join(too_small))
        warnings.warn(
            "fit_parameters: "
            + "; ".join(parts)
            + ". An optimiser such as Adam steps by about the learning rate in each "
            "coordinate's own units. For Parameter and float values the learning rate "
            "is a relative step (0.05 is about 5 %); for raw arrays it's absolute. "
            "Wrap physical values in Parameter, or change the learning rate.",
            UserWarning,
            stacklevel=3,
        )


# ── public API ───────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class FitResult:
    """Output of [`fit_parameters`][jbubble.fitting.fit_parameters].

    Attributes
    ----------
    params : PyTree
        Fitted parameters in physical units, with the structure of
        `params0`. Each [`Parameter`][jbubble.fitting.Parameter] and each
        fitted Python float is replaced by its fitted value, a `jax.Array`.
    loss_history : jax.Array, shape (n_accepted + 1,)
        `loss_history[0]` is the loss at `params0`, and `loss_history[k]` the
        loss after `k` accepted steps. `loss_history[-1]` is the loss at
        `params`.
    result : SimulationResult or list of SimulationResult
        The simulation at `params`. With `conditions`, a list with one
        [`SimulationResult`][jbubble.simulation.SimulationResult] per
        condition, in the same order.
    num_rejected : int
        Number of candidate steps rejected because a solve failed or the
        loss or gradient wasn't finite. Each rejection halves the step.
    stopped_early : bool
        `True` if the fit ended before `n_steps` accepted steps.
    message : str
        Why the fit ended.
    """

    params: PyTree
    loss_history: jax.Array
    result: SimulationResult | list[SimulationResult]
    num_rejected: int = 0
    stopped_early: bool = False
    message: str = ""


def fit_parameters(
    make_model: Callable[..., tuple[EquationOfMotion, Pulse]],
    params0: PyTree,
    *,
    loss_fn: Callable[..., ArrayLike],
    optimizer: optax.GradientTransformation,
    n_steps: int = 200,
    conditions: Sequence[PyTree] | None = None,
    save_spec: SaveSpec | None = None,
    t_max: ArrayLike | None = None,
    config: SolverConfig | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_backtracks: int = 5,
    step_callback: Callable[[int, PyTree, float], None] | None = None,
    log_every: int = 25,
) -> FitResult:
    r"""Fit model parameters by differentiating through the ODE solve.

    `fit_parameters` minimises the mean loss over the conditions,

    $$
    L(\theta) = \frac{1}{N} \sum_{c=1}^{N}
    \ell\big(\mathrm{simulate}(\mathrm{make\_model}(p, c)),\, c\big),
    \qquad p = \mathrm{unwrap}(\theta),
    $$

    where $\theta$ holds the optimiser's coordinates (see
    [`Parameter`][jbubble.fitting.Parameter]), $\ell$ is `loss_fn`, and
    $\mathrm{simulate}$ is
    [`run_simulation`][jbubble.simulation.run_simulation] with `save_spec`,
    `t_max`, `config`, and `adjoint`. Without `conditions`, $N = 1$ and
    neither `make_model` nor `loss_fn` receives a condition. Each step
    computes the update $\Delta\theta$ with `optimizer` and tries

    $$
    \theta_k = \theta + 2^{-k}\,\Delta\theta, \qquad k = 0, 1, \ldots,
    \texttt{max\_backtracks},
    $$

    accepting the first $\theta_k$ at which every solve converges and both
    $L$ and its gradient are finite.

    `fit_parameters` runs a Python loop around compiled functions, so you
    can't call it under `jax.jit` or `jax.vmap`, and each call compiles
    again. For many fits, such as multi-start fits, write the training loop
    yourself with [`unwrap`][jbubble.fitting.unwrap].

    Parameters
    ----------
    make_model : callable
        `make_model(params) -> (eom, pulse)`, or
        `make_model(params, condition) -> (eom, pulse)` when you pass
        `conditions`. `params` has the structure of `params0`, in physical
        units. Must be JAX-traceable.
    params0 : PyTree
        Initial parameters: a float, a `Parameter`, a dict, list, or tuple
        of them, or an Equinox module such as a
        [`NeuralProperty`][jbubble.bubble.property.NeuralProperty]. What
        gets fitted:

        - A `Parameter`, on its scaled coordinate, unless it's `fixed`.
        - A Python float (or `np.float64`) in `params0` itself or in a dict,
          list, or tuple: `1e-9` is shorthand for `Parameter(1e-9)`.
        - A floating-point JAX or NumPy array, including a NumPy scalar
          other than `np.float64`, in its own units, for example neural
          network weights.

        Everything else is held fixed: integers, booleans, strings,
        callables, and Python floats inside an Equinox module. If you set a
        Python float inside a module, `fit_parameters` warns and names it.
    loss_fn : callable
        `loss_fn(result) -> scalar`, or `loss_fn(result, condition) ->
        scalar` when you pass `conditions`. `result` is a
        [`SimulationResult`][jbubble.simulation.SimulationResult].
    optimizer : optax.GradientTransformation
        For example `optax.adam(0.05)`. For `Parameter` and float values,
        the learning rate is a relative step size: 0.05 changes a value by
        about 5 % per step. `optimizer.update` receives the current
        coordinates as `params`, so transformations such as `optax.adamw`
        work, but act on the coordinates: weight decay pulls a lower-bounded
        `Parameter` toward its initial value, not toward zero.
    n_steps : int
        Number of accepted steps. Default: `200`.
    conditions : list or tuple, optional
        One entry per experimental condition, for example per driving
        pressure or per recording. Each entry can be any pytree, and
        `fit_parameters` passes it to `make_model` and `loss_fn`. All
        conditions share `params`. Conditions with the same structure and
        leaf shapes run in parallel under `jax.vmap`; others run one after
        another. In parallel, each number in a condition, including a
        Python int, reaches `make_model` and `loss_fn` as a traced array:
        use it in `jnp.where` or `jax.lax.cond`, not in an `if` or as a
        slice bound. A condition that holds a Python bool or a string runs
        one after another, so these stay Python values that you can use as
        flags.
    save_spec : SaveSpec, optional
        Output sampling. `None` uses [`SaveSpec()`][jbubble.solver.SaveSpec].
    t_max : float, optional
        Integration end time [s]. `None` uses each pulse's `t_end`.
    config : SolverConfig, optional
        ODE solver settings. `None` uses
        [`SolverConfig()`][jbubble.solver.SolverConfig], the same as
        [`run_simulation`][jbubble.simulation.run_simulation], so the model
        you fit is the model you simulate.
    adjoint : diffrax.AbstractAdjoint, optional
        How the gradient is computed through the solve. `None` uses
        `diffrax.RecursiveCheckpointAdjoint()`. `fit_parameters`
        differentiates in reverse mode, so it can't use
        `diffrax.ForwardMode()`.
    max_backtracks : int
        How many times to halve a step that fails before stopping. If every
        halved step fails, the fit warns and returns the last accepted
        parameters. Default: `5`.
    step_callback : callable, optional
        `step_callback(step, params, loss)`, called outside JIT for
        `params0` (`step == 0`) and after each accepted step. `params` is in
        physical units, and `loss` is the loss at `params`. Raise
        `StopIteration` to end the fit. Any other exception propagates and
        discards the fit, so save checkpoints from the callback if you need
        them.
    log_every : int
        Print progress every `log_every` steps. `0` turns off logging.
        Default: `25`.

    Returns
    -------
    FitResult
        Fitted parameters, loss history, the simulation at the fitted
        parameters, and how the fit ended.

    Raises
    ------
    ValueError
        If `params0` has nothing to fit, a Python float in `params0` is zero,
        or `conditions` is empty.
    TypeError
        If `adjoint` is `diffrax.ForwardMode()`, `conditions` isn't a list
        or tuple, `make_model` or `loss_fn` takes the wrong number of
        arguments, `make_model` doesn't return an `(eom, pulse)` tuple, or
        `loss_fn` doesn't return a scalar.
    RuntimeError
        If a solve fails, or the loss or its gradient isn't finite, at
        `params0`.

    Warns
    -----
    UserWarning
        If you set a Python float inside an Equinox module in `params0`,
        which stays fixed, or if the first step is larger than a
        coordinate's scale or smaller than a millionth of it.
    RuntimeWarning
        If every halved step fails and the fit stops early.

    Examples
    --------
    ```python
    import optax
    from jbubble import fit_parameters
    from jbubble.fitting import Parameter
    from jbubble.metrics import normalised_mse_radius

    fit = fit_parameters(
        lambda p: (make_eom(kappa_s=p["kappa_s"], chi=p["chi"]), pulse),
        {
            "kappa_s": Parameter(1e-9, lower=0.0),  # [N s/m]
            "chi": Parameter(0.3, lower=0.0, upper=1.5),  # [N/m]
        },
        loss_fn=lambda r: normalised_mse_radius(r.radius, measured, R0),
        optimizer=optax.adam(0.05),
    )
    ```
    """
    config = SolverConfig() if config is None else config
    adjoint = diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
    if isinstance(adjoint, diffrax.ForwardMode):
        raise TypeError(
            "fit_parameters: diffrax.ForwardMode() gives only forward-mode "
            "derivatives, and fit_parameters differentiates in reverse mode. Keep "
            "the default diffrax.RecursiveCheckpointAdjoint(). For a least-squares "
            "fit with forward-mode Jacobians, pass adjoint=diffrax.ForwardMode() "
            "to run_simulation instead, as the fitting guide shows."
        )

    if conditions is not None:
        if not isinstance(conditions, (list, tuple)):
            raise TypeError(
                "fit_parameters: conditions must be a list or tuple with one entry "
                "per condition, for example [{'pressure': 50e3, 'radius': r50}, ...]; "
                f"got {type(conditions).__name__}."
            )
        if not conditions:
            raise ValueError("fit_parameters: conditions is empty.")
        conditions = list(conditions)
    for fn, name, first in (
        (make_model, "make_model", "params"),
        (loss_fn, "loss_fn", "result"),
    ):
        _check_arity(fn, name, (first,) if conditions is None else (first, "condition"))
    stacked = None if conditions is None else _stack(conditions)

    spec = _canonicalise(params0)
    _warn_held_floats(spec)
    trainable, static = eqx.partition(spec, _trainable_filter(spec))
    # Strong dtypes: a weakly typed array, such as jnp.array(1e-9), comes back
    # strongly typed from the first update and would make `evaluate` compile
    # a second time.
    trainable = jtu.tree_map(
        lambda x: jnp.asarray(x, dtype=jnp.result_type(x)), trainable
    )
    if not jtu.tree_leaves(trainable):
        raise ValueError(
            "fit_parameters: params0 has nothing to fit. Use Parameter or Python "
            "floats for physical values, and floating-point arrays for network "
            "weights. Integers, booleans, fixed Parameters, and Python floats "
            "inside an Equinox module are held fixed."
        )

    def one(params: PyTree, condition: Any) -> tuple[jax.Array, SimulationResult]:
        model = (
            make_model(params) if conditions is None else make_model(params, condition)
        )
        if not _is_model(model):
            raise TypeError(
                "fit_parameters: make_model must return an (EquationOfMotion, Pulse) "
                f"tuple; got {type(model).__name__}."
            )
        result = _simulate(
            model[0],
            model[1],
            save_spec=save_spec,
            t_max=t_max,
            config=config,
            adjoint=adjoint,
        )
        loss = jnp.asarray(
            loss_fn(result) if conditions is None else loss_fn(result, condition)
        )
        if loss.ndim != 0:
            raise TypeError(
                f"fit_parameters: loss_fn must return a scalar; got shape {loss.shape}."
            )
        return loss, result

    def objective(tr: PyTree, st: PyTree, conds: Any):
        params = unwrap(eqx.combine(tr, st))
        if conditions is None:
            loss, result = one(params, None)
            return loss, (result, result.converged[None], loss[None])
        if stacked is not None:
            losses, results = eqx.filter_vmap(lambda c: one(params, c))(conds)
            return jnp.mean(losses), (results, results.converged, losses)
        outs = [one(params, c) for c in conds]
        losses = jnp.stack([loss for loss, _ in outs])
        results = [r for _, r in outs]
        converged = jnp.stack([r.converged for r in results])
        return jnp.mean(losses), (results, converged, losses)

    @eqx.filter_jit
    def evaluate(tr: PyTree, st: PyTree, conds: Any):
        (loss, (results, converged, losses)), grads = jax.value_and_grad(
            objective, has_aux=True
        )(tr, st, conds)
        grads_finite = _all_finite(grads)
        ok = jnp.all(converged) & jnp.isfinite(loss) & grads_finite
        return loss, grads, ok, (converged, losses, grads_finite), results

    @eqx.filter_jit
    def propose(tr: PyTree, opt_state: Any, grads: PyTree):
        return optimizer.update(grads, opt_state, tr)

    @eqx.filter_jit
    def apply(tr: PyTree, updates: PyTree, factor: jax.Array):
        return eqx.apply_updates(tr, jtu.tree_map(lambda u: factor * u, updates))

    conds_arg = stacked if stacked is not None else conditions
    loss, grads, ok, diagnostics, results = evaluate(trainable, static, conds_arg)
    if not bool(ok):
        converged, losses, grads_finite = diagnostics
        problems = []
        for i, (ok_i, loss_i) in enumerate(
            zip(converged.tolist(), losses.tolist(), strict=True)
        ):
            where = "" if conditions is None else f"condition {i}: "
            if not ok_i:
                problems.append(f"{where}the ODE solve did not converge")
            elif not jnp.isfinite(loss_i):
                problems.append(f"{where}the loss is {loss_i}")
        if not problems and not bool(grads_finite):
            problems.append("the gradient is not finite")
        if not all(converged.tolist()):
            hint = (
                "Check params0 and the driving pressure, raise "
                "SolverConfig.max_steps, or tighten the solver tolerances."
            )
        else:
            hint = (
                "The solve converged, so check that loss_fn returns a finite "
                "value with a finite gradient at params0."
            )
        raise RuntimeError(
            "fit_parameters: can't start at params0, because "
            + "; ".join(problems)
            + ". "
            + hint
        )

    def physical(tr: PyTree) -> PyTree:
        return unwrap(eqx.combine(tr, static))

    # Typical magnitude of each coordinate, for the first-step check.
    reference = eqx.partition(
        jtu.tree_map(
            lambda x: (
                eqx.tree_at(lambda q: q.raw, x, jnp.ones_like(x.raw))
                if _is_parameter(x)
                else x
            ),
            spec,
            is_leaf=_is_parameter,
        ),
        _trainable_filter(spec),
    )[0]
    opt_state = optimizer.init(trainable)
    history = [float(loss)]
    num_rejected = 0
    stopped_early = False
    message = f"completed {n_steps} steps"

    def report(step: int) -> None:
        if log_every > 0 and (step % log_every == 0 or step == n_steps):
            print(
                f"  step {step:>4} / {n_steps}  loss = {history[-1]:.4e}  "
                f"{_format_params(physical(trainable))}"
                + (f"  (rejected {num_rejected})" if num_rejected else "")
            )

    def callback(step: int) -> bool:
        if step_callback is None:
            return False
        try:
            step_callback(step, physical(trainable), history[-1])
        except StopIteration:
            return True
        return False

    report(0)
    stop = callback(0)
    if stop:
        stopped_early = n_steps > 0
        message = "step_callback raised StopIteration at step 0"
    step = 0
    while not stop and step < n_steps:
        updates, new_opt_state = propose(trainable, opt_state, grads)
        if step == 0:
            _warn_on_first_step(reference, updates)
        for k in range(max_backtracks + 1):
            candidate = apply(trainable, updates, jnp.asarray(0.5**k))
            c_loss, c_grads, c_ok, _, c_results = evaluate(candidate, static, conds_arg)
            if bool(c_ok):
                break
            num_rejected += 1
        else:
            stopped_early = True
            message = (
                f"stopped after {step} of {n_steps} steps: {max_backtracks + 1} "
                "successively halved steps all made a solve fail or gave a non-finite "
                "loss or gradient"
            )
            warnings.warn(
                f"fit_parameters: {message}. Returning the last accepted parameters. "
                "Lower the learning rate, bound the parameters with Parameter, or "
                "raise SolverConfig.max_steps.",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        trainable, opt_state = candidate, new_opt_state
        loss, grads, results = c_loss, c_grads, c_results
        step += 1
        history.append(float(loss))
        report(step)
        stop = callback(step)
        if stop:
            stopped_early = step < n_steps
            message = f"step_callback raised StopIteration after step {step}"

    if stacked is not None:
        num_conditions = len(jtu.tree_leaves(stacked)[0])
        results = [
            jtu.tree_map(lambda x, i=i: x[i], results) for i in range(num_conditions)
        ]

    return FitResult(
        params=physical(trainable),
        loss_history=jnp.asarray(history),
        result=results,
        num_rejected=num_rejected,
        stopped_early=stopped_early,
        message=message,
    )
