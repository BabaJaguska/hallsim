"""PathwayMapper — multi-input Hill-function readout from mechanistic state to
ssGSEA-style pathway activity scores.

Bridges the gap between HallSim's mechanistic observables (p53, mTORC1, NF-κB,
ROS, ATP) and pathway-level activities reported by single-sample GSEA on
transcriptomics. Designed for direct comparison with experimental ssGSEA
normalized enrichment scores (NES).

Architecture
------------
``PathwayMapper`` is an :class:`equinox.Module`, NOT a :class:`Process`. It has
no own dynamics — pathway activity is a *readout* of the current state, not a
state variable that evolves. Making it a Module gives us:

- Tunable Hill parameters as JAX arrays → differentiable end-to-end →
  parameter calibration via ``jax.grad`` + ``optax``
- JIT-compilation of the whole mapping
- Native batch support (state of shape ``(n_time, batch, n_vars)``
  produces pathway scores of shape ``(n_time, batch)``)
- Composition with the rest of HallSim without forcing a fake derivative

Pathway formulas
----------------
For each input ``x`` (clamped non-negative), Hill activation and inhibition
are::

    H_act(x; K, n)   = x^n / (K^n + x^n)         # sigmoidal switch ON
    H_inhib(x; K, n) = K^n / (K^n + x^n)         # sigmoidal switch OFF

Pathway scores combine these::

    autophagy   = H_act(p53) · H_inhib(mTOR) · H_inhib(NF-κB)
                · H_act(ROS) · H_inhib(ATP)
    glycolysis  = H_act(mTOR) · H_act(NF-κB) · H_inhib(p53)
                · H_inhib(ATP) · H_inhib(ROS)
    senescence  = H_inhib(ATP) · weighted_avg(
                      H_act(p53), H_act(mTOR),
                      H_act(NF-κB), H_act(ROS))
    mtorc1_sig  = H_act(mTOR)
    nfkb_sig    = H_act(NF-κB)
    oxphos      = H_act(ATP)
    ros_pathway = H_act(ROS)

The first three encode multi-input regulatory logic from the literature
(Mathiassen 2017, Humpton 2016, Fielder 2017, Reed 2022). The last four are
proxy mappings from the directly relevant mechanistic state.

Why these formulas
------------------
- *Autophagy* is suppressed by mTORC1 (ULK1 phosphorylation), driven by p53
  (Sestrins/DRAM), inhibited by chronic NF-κB (autophagy/inflammation
  reciprocal regulation), driven by ROS (oxidative stress), and inhibited by
  high ATP (AMPK quiescent at high energy).
- *Glycolysis* is driven by mTORC1 (HIF-1α) and NF-κB (chronic inflammation
  → Warburg-like shift), inhibited by p53 (TIGAR axis), inhibited by ATP
  (energy charge), and inhibited by ROS (oxidative damage to glycolytic
  enzymes).
- *Senescence* requires energy depletion (H_inhib(ATP)) and combined
  stress (any of p53, mTOR, NF-κB, ROS active).

Default Hill parameters
-----------------------
Default ``K=0.5, n=4.0`` reflects the calibration result from the kosmos
discovery report (``docs/kosmos_Hallmarks1_edison.pdf``, page 9): a focused
calibration moved Pearson concordance with GSE248823 from ``r=0.79`` to
``r=0.99``, with ``n≈4`` cooperative Hill exponents. Keep defaults loose;
calibrate per dataset via :func:`calibrate_pathway_mapper`.

Usage
-----
>>> from hallsim.pathway_mapper import PathwayMapper
>>> mapper = PathwayMapper()
>>> scores = mapper.score(p53=0.8, mtor=0.6, nfkb=0.4, ros=0.3, atp=2.0)
>>> scores.autophagy, scores.senescence
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp


# ── Hill primitives ─────────────────────────────────────────────────────


def h_act(x: jnp.ndarray, K: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """Hill activation: ``x^n / (K^n + x^n)``. Bounded in [0, 1].

    Inputs are clamped non-negative for numerical safety — Hill activation
    is biophysically defined only for non-negative concentrations / activity.
    """
    x_pos = jnp.maximum(x, 0.0)
    x_n = x_pos**n
    K_n = K**n
    return x_n / (K_n + x_n + 1e-12)


def h_inhib(x: jnp.ndarray, K: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """Hill inhibition: ``K^n / (K^n + x^n)``. Bounded in [0, 1]. Inverse of
    :func:`h_act`."""
    x_pos = jnp.maximum(x, 0.0)
    return 1.0 - h_act(x_pos, K, n)


# ── Output container ───────────────────────────────────────────────────


@dataclass(frozen=True)
class PathwayScores:
    """Seven pathway activity scores, each in [0, 1].

    Shape-polymorphic: scalar inputs → scalar fields; ``(batch,)`` inputs →
    ``(batch,)`` fields; ``(n_time, batch)`` inputs → ``(n_time, batch)``
    fields. The trailing-axis convention from the rest of HallSim is preserved.
    """

    autophagy: jnp.ndarray
    glycolysis: jnp.ndarray
    senescence: jnp.ndarray
    mtorc1_signaling: jnp.ndarray
    nfkb_signaling: jnp.ndarray
    oxphos: jnp.ndarray
    ros_pathway: jnp.ndarray

    def as_dict(self) -> dict[str, jnp.ndarray]:
        """Return scores as ``{pathway_name: score}``. Useful for vector ops
        (Pearson correlation) against experimental ssGSEA dicts."""
        return {
            "autophagy": self.autophagy,
            "glycolysis": self.glycolysis,
            "senescence": self.senescence,
            "mtorc1_signaling": self.mtorc1_signaling,
            "nfkb_signaling": self.nfkb_signaling,
            "oxphos": self.oxphos,
            "ros_pathway": self.ros_pathway,
        }

    def as_vector(self, names: tuple[str, ...] | None = None) -> jnp.ndarray:
        """Stack scores along a new trailing axis in the given name order. The
        name order pins how this vector aligns with experimental ssGSEA
        deltas — keep it consistent across simulated and measured sides."""
        if names is None:
            names = PATHWAY_ORDER
        d = self.as_dict()
        return jnp.stack([d[n] for n in names], axis=-1)


PATHWAY_ORDER: tuple[str, ...] = (
    "autophagy",
    "glycolysis",
    "senescence",
    "mtorc1_signaling",
    "nfkb_signaling",
    "oxphos",
    "ros_pathway",
)


# ── PathwayMapper ──────────────────────────────────────────────────────


class PathwayMapper(eqx.Module):
    """Multi-input Hill-function readout layer.

    Parameters
    ----------
    K_*:
        Half-saturation constants for each input ``(p53, mtor, nfkb, ros,
        atp)``. Default 0.5 places the inflection at half the typical
        active range of each observable.
    n_*:
        Hill cooperativity exponents. Default 4.0 — calibrated from the
        kosmos result; gives switch-like behavior typical of
        multi-site phosphorylation cascades.
    w_sen_*:
        Weights for the senescence weighted-average across the four stress
        axes (p53, mTOR, NF-κB, ROS). Defaults equal to 1.0 each
        (normalized to sum to 1 internally).

    All parameters are JAX arrays — differentiable, JIT-friendly,
    pytree-compatible. Use ``eqx.tree_at`` to construct calibrated
    instances; see :func:`calibrate_pathway_mapper`.
    """

    K_p53: jnp.ndarray
    K_mtor: jnp.ndarray
    K_nfkb: jnp.ndarray
    K_ros: jnp.ndarray
    K_atp: jnp.ndarray

    n_p53: jnp.ndarray
    n_mtor: jnp.ndarray
    n_nfkb: jnp.ndarray
    n_ros: jnp.ndarray
    n_atp: jnp.ndarray

    w_sen_p53: jnp.ndarray
    w_sen_mtor: jnp.ndarray
    w_sen_nfkb: jnp.ndarray
    w_sen_ros: jnp.ndarray

    def __init__(
        self,
        *,
        K: float = 0.5,
        n: float = 4.0,
        K_atp: float | None = None,
        n_atp: float | None = None,
    ) -> None:
        # Per-input Hill K/n. Defaults are uniform; ATP can be overridden
        # because its scale (mfunct + glycolysis ~ 6) differs from the
        # bounded activities (~ 0–2).
        self.K_p53 = jnp.asarray(K)
        self.K_mtor = jnp.asarray(K)
        self.K_nfkb = jnp.asarray(K)
        self.K_ros = jnp.asarray(K)
        self.K_atp = jnp.asarray(K_atp if K_atp is not None else 3.0)

        self.n_p53 = jnp.asarray(n)
        self.n_mtor = jnp.asarray(n)
        self.n_nfkb = jnp.asarray(n)
        self.n_ros = jnp.asarray(n)
        self.n_atp = jnp.asarray(n_atp if n_atp is not None else n)

        # Equal weights for the senescence weighted-average; calibration
        # can shift them.
        self.w_sen_p53 = jnp.asarray(1.0)
        self.w_sen_mtor = jnp.asarray(1.0)
        self.w_sen_nfkb = jnp.asarray(1.0)
        self.w_sen_ros = jnp.asarray(1.0)

    # ── Core scoring ───────────────────────────────────────────────

    def score(
        self,
        *,
        p53: jnp.ndarray,
        mtor: jnp.ndarray,
        nfkb: jnp.ndarray,
        ros: jnp.ndarray,
        atp: jnp.ndarray,
    ) -> PathwayScores:
        """Compute seven pathway scores from five mechanistic observables.

        All inputs are scalars or arrays of identical shape. Output fields
        share that shape. JAX-traceable end-to-end.
        """
        H_p53_act = h_act(p53, self.K_p53, self.n_p53)
        H_p53_inh = h_inhib(p53, self.K_p53, self.n_p53)
        H_mtor_act = h_act(mtor, self.K_mtor, self.n_mtor)
        H_mtor_inh = h_inhib(mtor, self.K_mtor, self.n_mtor)
        H_nfkb_act = h_act(nfkb, self.K_nfkb, self.n_nfkb)
        H_nfkb_inh = h_inhib(nfkb, self.K_nfkb, self.n_nfkb)
        H_ros_act = h_act(ros, self.K_ros, self.n_ros)
        H_ros_inh = h_inhib(ros, self.K_ros, self.n_ros)
        H_atp_act = h_act(atp, self.K_atp, self.n_atp)
        H_atp_inh = h_inhib(atp, self.K_atp, self.n_atp)

        # Multi-input pathway formulas (kosmos page 8).
        autophagy = H_p53_act * H_mtor_inh * H_nfkb_inh * H_ros_act * H_atp_inh
        glycolysis = (
            H_mtor_act * H_nfkb_act * H_p53_inh * H_atp_inh * H_ros_inh
        )

        # Senescence: stress-axis weighted average gated by low ATP.
        w_sum = (
            self.w_sen_p53
            + self.w_sen_mtor
            + self.w_sen_nfkb
            + self.w_sen_ros
            + 1e-12
        )
        stress_avg = (
            self.w_sen_p53 * H_p53_act
            + self.w_sen_mtor * H_mtor_act
            + self.w_sen_nfkb * H_nfkb_act
            + self.w_sen_ros * H_ros_act
        ) / w_sum
        senescence = H_atp_inh * stress_avg

        return PathwayScores(
            autophagy=autophagy,
            glycolysis=glycolysis,
            senescence=senescence,
            mtorc1_signaling=H_mtor_act,
            nfkb_signaling=H_nfkb_act,
            oxphos=H_atp_act,
            ros_pathway=H_ros_act,
        )

    # ── Convenience: from ERiQ trajectory ───────────────────────────

    def from_eriq_state(
        self,
        state: dict[str, jnp.ndarray],
        *,
        prefix: str = "eriq",
    ) -> PathwayScores:
        """Map a HallSim ERiQ state dict directly to pathway scores.

        Pulls ``p53_activity``, ``mTOR_activity``, ``ROS_activity``,
        ``mito_function``, ``glycolysis`` from store paths
        ``{prefix}/<name>``, then computes the algebraic NF-κB and total
        ATP using the same formulas as :func:`hallsim.models.eriq._compute_algebraic`,
        and finally calls :meth:`score`.

        Use this on a single time slice
        (``composite.unflatten(result.ys[t])``) or on full trajectory
        slices via ``jax.vmap``.
        """
        from hallsim.models.eriq import _compute_algebraic

        p = prefix
        p53 = state[f"{p}/p53_activity"]
        # _compute_algebraic expects un-prefixed port names, mimic the
        # per-process port view that ERiQ Processes get.
        sub = {
            "mito_function": state[f"{p}/mito_function"],
            "glycolysis": state[f"{p}/glycolysis"],
            "mito_damage": state[f"{p}/mito_damage"],
            "mTOR_activity": state[f"{p}/mTOR_activity"],
            "p53_activity": p53,
            "ROS_activity": state[f"{p}/ROS_activity"],
            "ROS_integrator_c": state[f"{p}/ROS_integrator_c"],
        }
        obs = _compute_algebraic(sub)
        return self.score(
            p53=p53,
            mtor=obs["MTOR"],
            nfkb=obs["NFKB"],
            ros=obs["ROS"],
            atp=obs["ATPr"],
        )

    def from_eriq_trajectory(
        self,
        result,
        *,
        prefix: str = "eriq",
    ) -> PathwayScores:
        """Map an entire ``SchedulerResult`` to time-series pathway scores.

        ``result.ys`` has shape ``(n_time, ..., n_vars)``. The output
        ``PathwayScores`` fields share the leading shape ``(n_time, ...)``.
        """
        # Pull the trajectories by store path; each is (n_time, ...).
        p = prefix
        p53_t = result.get(f"{p}/p53_activity")
        mfunct_t = result.get(f"{p}/mito_function")
        gly_t = result.get(f"{p}/glycolysis")
        mdam_t = result.get(f"{p}/mito_damage")
        mtor_t = result.get(f"{p}/mTOR_activity")
        ros_t = result.get(f"{p}/ROS_activity")
        ros_int_t = result.get(f"{p}/ROS_integrator_c")

        # Vectorize the algebraic computation over all (n_time, ...) shapes.
        # _compute_algebraic is built on jnp ops, so it broadcasts naturally
        # — no vmap required.
        from hallsim.models.eriq import _compute_algebraic

        sub = {
            "mito_function": mfunct_t,
            "glycolysis": gly_t,
            "mito_damage": mdam_t,
            "mTOR_activity": mtor_t,
            "p53_activity": p53_t,
            "ROS_activity": ros_t,
            "ROS_integrator_c": ros_int_t,
        }
        obs = _compute_algebraic(sub)
        return self.score(
            p53=p53_t,
            mtor=obs["MTOR"],
            nfkb=obs["NFKB"],
            ros=obs["ROS"],
            atp=obs["ATPr"],
        )


# ── Concordance metric ─────────────────────────────────────────────────


def pearson_r(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Pearson correlation between two 1-D vectors. Returns a scalar
    ``jnp.ndarray`` so it composes inside JIT/grad."""
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    num = jnp.sum(a * b)
    den = jnp.sqrt(jnp.sum(a**2) * jnp.sum(b**2)) + 1e-12
    return num / den


def sign_agreement(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Fraction of entries where ``a`` and ``b`` have the same sign. Floats
    are compared against zero. Returns a scalar in [0, 1]."""
    return jnp.mean(jnp.sign(a) == jnp.sign(b))


# ── Calibration ───────────────────────────────────────────────────────


def calibrate_pathway_mapper(
    *,
    delta_sim_inputs: dict[str, jnp.ndarray],
    delta_data: jnp.ndarray,
    initial: PathwayMapper | None = None,
    n_steps: int = 1000,
    learning_rate: float = 0.05,
    pathway_order: tuple[str, ...] = PATHWAY_ORDER,
) -> tuple[PathwayMapper, dict]:
    """Fit Hill parameters to maximize Pearson r between Δ_sim and Δ_data.

    Treats *the simulator inputs as fixed* and tunes only the mapper. The
    intuition (kosmos page 9): when most of the discrepancy between
    simulated and measured pathway changes comes from interpretation-layer
    Hill parameters rather than mechanism, mapper calibration alone
    recovers strong concordance.

    Parameters
    ----------
    delta_sim_inputs:
        ``{"p53": (perturbed - control) ..., "mtor": ..., "nfkb": ...,
        "ros": ..., "atp": ...}`` — perturbed-minus-control simulator
        observable deltas. The mapper is applied to *control* and
        *perturbed* observable values via finite differences here, so we
        require both: pass under keys
        ``{"p53_ctrl", "p53_pert", "mtor_ctrl", ...}``.
    delta_data:
        Reference Δ_data vector aligned to ``pathway_order``, length 7.
    initial:
        Starting PathwayMapper; defaults to the un-calibrated K=0.5, n=4.
    n_steps, learning_rate:
        Optimizer hyperparameters.
    pathway_order:
        Pathway name order to align simulated and measured vectors.

    Returns
    -------
    ``(calibrated_mapper, history)`` — the fitted PathwayMapper and a dict
    of per-step ``{"loss": ..., "r": ...}`` arrays.
    """
    import optax

    if initial is None:
        initial = PathwayMapper()

    delta_data = jnp.asarray(delta_data)

    def delta_sim(mapper: PathwayMapper) -> jnp.ndarray:
        s_ctrl = mapper.score(
            p53=delta_sim_inputs["p53_ctrl"],
            mtor=delta_sim_inputs["mtor_ctrl"],
            nfkb=delta_sim_inputs["nfkb_ctrl"],
            ros=delta_sim_inputs["ros_ctrl"],
            atp=delta_sim_inputs["atp_ctrl"],
        ).as_vector(pathway_order)
        s_pert = mapper.score(
            p53=delta_sim_inputs["p53_pert"],
            mtor=delta_sim_inputs["mtor_pert"],
            nfkb=delta_sim_inputs["nfkb_pert"],
            ros=delta_sim_inputs["ros_pert"],
            atp=delta_sim_inputs["atp_pert"],
        ).as_vector(pathway_order)
        return s_pert - s_ctrl

    def loss_fn(mapper: PathwayMapper) -> jnp.ndarray:
        d = delta_sim(mapper)
        # Maximize Pearson r ⇒ minimize (1 - r). Add a small L2 penalty
        # to keep n away from runaway values.
        r = pearson_r(d, delta_data)
        params = jnp.stack(
            [
                mapper.n_p53,
                mapper.n_mtor,
                mapper.n_nfkb,
                mapper.n_ros,
                mapper.n_atp,
            ]
        )
        l2_n = 1e-3 * jnp.sum((params - 4.0) ** 2)
        return (1.0 - r) + l2_n

    # Split params (trainable) from static state (none here).
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(initial, eqx.is_array))

    @eqx.filter_jit
    def step(mapper, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(mapper)
        updates, opt_state = optimizer.update(grads, opt_state, mapper)
        mapper = eqx.apply_updates(mapper, updates)
        return mapper, opt_state, loss

    mapper = initial
    losses, rs = [], []
    for _ in range(n_steps):
        mapper, opt_state, loss = step(mapper, opt_state)
        losses.append(float(loss))
        rs.append(float(pearson_r(delta_sim(mapper), delta_data)))
        # Keep n ≥ 1 (cooperativity must be positive); clamp via tree_at.
        mapper = _clamp_hill_params(mapper)

    history = {"loss": jnp.asarray(losses), "r": jnp.asarray(rs)}
    return mapper, history


def _clamp_hill_params(mapper: PathwayMapper) -> PathwayMapper:
    """Keep K > 0 and n ≥ 1 — outside this regime the Hill becomes
    singular or non-cooperative. Calibration occasionally pushes into
    these zones; clamp after each step."""

    def clamp_pos(x):
        return jnp.maximum(x, 1e-3)

    def clamp_n(x):
        return jnp.maximum(x, 1.0)

    return eqx.tree_at(
        lambda m: (
            m.K_p53,
            m.K_mtor,
            m.K_nfkb,
            m.K_ros,
            m.K_atp,
            m.n_p53,
            m.n_mtor,
            m.n_nfkb,
            m.n_ros,
            m.n_atp,
        ),
        mapper,
        (
            clamp_pos(mapper.K_p53),
            clamp_pos(mapper.K_mtor),
            clamp_pos(mapper.K_nfkb),
            clamp_pos(mapper.K_ros),
            clamp_pos(mapper.K_atp),
            clamp_n(mapper.n_p53),
            clamp_n(mapper.n_mtor),
            clamp_n(mapper.n_nfkb),
            clamp_n(mapper.n_ros),
            clamp_n(mapper.n_atp),
        ),
    )
