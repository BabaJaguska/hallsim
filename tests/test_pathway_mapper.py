"""Tests for PathwayMapper — Hill primitives, pathway formulas, scenario
consistency, batching, differentiability, and calibration convergence.

Validation strategy: every claim the module makes about its outputs is a
testable invariant (bounds, monotonicity, dose-response). Calibration is
tested on a synthetic ground-truth scenario where we know the answer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hallsim.pathway_mapper import (
    PATHWAY_ORDER,
    PathwayMapper,
    calibrate_pathway_mapper,
    h_act,
    h_inhib,
    pearson_r,
    sign_agreement,
)


# ═══════════════════════════════════════════════════════════════════════════
# Hill primitives
# ═══════════════════════════════════════════════════════════════════════════


class TestHillPrimitives:
    """``h_act`` and ``h_inhib`` are the building blocks — verify the
    invariants ssGSEA-style mapping relies on."""

    @pytest.mark.parametrize("x", [0.0, 0.1, 0.5, 1.0, 5.0, 100.0])
    def test_h_act_in_unit_interval(self, x):
        v = float(h_act(jnp.asarray(x), jnp.asarray(0.5), jnp.asarray(4.0)))
        assert 0.0 <= v <= 1.0

    @pytest.mark.parametrize("x", [0.0, 0.1, 0.5, 1.0, 5.0, 100.0])
    def test_h_inhib_in_unit_interval(self, x):
        v = float(h_inhib(jnp.asarray(x), jnp.asarray(0.5), jnp.asarray(4.0)))
        assert 0.0 <= v <= 1.0

    def test_act_plus_inhib_is_one(self):
        x = jnp.asarray([0.1, 0.5, 1.0, 2.0, 5.0])
        K = jnp.asarray(0.5)
        n = jnp.asarray(4.0)
        assert jnp.allclose(h_act(x, K, n) + h_inhib(x, K, n), 1.0, atol=1e-6)

    def test_h_act_at_K_equals_half(self):
        """Hill activation crosses 0.5 at x=K — definitional."""
        K = jnp.asarray(0.7)
        n = jnp.asarray(3.0)
        assert jnp.isclose(h_act(K, K, n), 0.5, atol=1e-6)

    def test_h_act_negative_input_clamped(self):
        """Negative concentrations clamp to zero — Hill is biophysically
        defined only for non-negative inputs."""
        v = float(h_act(jnp.asarray(-5.0), jnp.asarray(0.5), jnp.asarray(2.0)))
        assert v == 0.0

    def test_h_act_monotone_increasing(self):
        K = jnp.asarray(0.5)
        n = jnp.asarray(4.0)
        xs = jnp.linspace(0.01, 10.0, 50)
        ys = h_act(xs, K, n)
        diffs = jnp.diff(ys)
        assert jnp.all(diffs >= -1e-7)

    def test_h_inhib_monotone_decreasing(self):
        K = jnp.asarray(0.5)
        n = jnp.asarray(4.0)
        xs = jnp.linspace(0.01, 10.0, 50)
        ys = h_inhib(xs, K, n)
        diffs = jnp.diff(ys)
        assert jnp.all(diffs <= 1e-7)


# ═══════════════════════════════════════════════════════════════════════════
# PathwayMapper output shape and bounds
# ═══════════════════════════════════════════════════════════════════════════


class TestPathwayMapperBounds:

    def test_output_pathway_count(self):
        assert len(PATHWAY_ORDER) == 7

    def test_all_scores_in_unit_interval(self):
        mapper = PathwayMapper()
        # Sweep a grid of inputs; every output must stay in [0, 1].
        for p53 in [0.05, 0.5, 2.0]:
            for mtor in [0.05, 0.5, 2.0]:
                for atp in [0.5, 2.0, 5.0]:
                    scores = mapper.score(
                        p53=jnp.asarray(p53),
                        mtor=jnp.asarray(mtor),
                        nfkb=jnp.asarray(0.5),
                        ros=jnp.asarray(0.5),
                        atp=jnp.asarray(atp),
                    )
                    for v in scores.as_dict().values():
                        v = float(v)
                        assert 0.0 <= v <= 1.0

    def test_as_vector_pathway_order(self):
        mapper = PathwayMapper()
        scores = mapper.score(p53=0.5, mtor=0.5, nfkb=0.5, ros=0.5, atp=2.0)
        v = scores.as_vector()
        assert v.shape == (7,)
        assert v.shape[-1] == len(PATHWAY_ORDER)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario consistency — the mapper should give biologically sane outputs in
# canonical scenarios. These are the smoke tests for "is the multi-input
# logic wired correctly?"
# ═══════════════════════════════════════════════════════════════════════════


class TestScenarioConsistency:

    def test_high_p53_low_mtor_drives_autophagy(self):
        """p53-active + mTORC1-inhibited + low NF-κB should give high
        autophagy score (the canonical autophagy-induction state)."""
        mapper = PathwayMapper(K=0.5, n=4.0, K_atp=3.0)
        scores = mapper.score(p53=2.0, mtor=0.05, nfkb=0.05, ros=0.5, atp=1.0)
        assert float(scores.autophagy) > 0.4

    def test_high_mtor_high_nfkb_low_p53_drives_glycolysis(self):
        """Anabolic + inflamed state → glycolytic shift."""
        mapper = PathwayMapper(K=0.5, n=4.0, K_atp=3.0)
        scores = mapper.score(p53=0.05, mtor=2.0, nfkb=2.0, ros=0.05, atp=1.0)
        assert float(scores.glycolysis) > 0.3

    def test_low_atp_high_stress_drives_senescence(self):
        """Energy depletion + multi-axis stress → senescence."""
        mapper = PathwayMapper(K=0.5, n=4.0, K_atp=3.0)
        scores = mapper.score(p53=2.0, mtor=2.0, nfkb=2.0, ros=2.0, atp=0.5)
        assert float(scores.senescence) > 0.4

    def test_high_atp_low_stress_low_senescence(self):
        """Healthy state → near-zero senescence."""
        mapper = PathwayMapper(K=0.5, n=4.0, K_atp=3.0)
        scores = mapper.score(
            p53=0.05, mtor=0.05, nfkb=0.05, ros=0.05, atp=5.0
        )
        assert float(scores.senescence) < 0.1

    def test_ddis_proxy_directional(self):
        """DDIS-like state (high p53, high ROS, low ATP) vs control —
        senescence and autophagy should rise; oxphos should fall."""
        mapper = PathwayMapper(K=0.5, n=4.0, K_atp=3.0)
        ctrl = mapper.score(p53=0.2, mtor=0.5, nfkb=0.3, ros=0.2, atp=4.0)
        ddis = mapper.score(p53=1.5, mtor=0.5, nfkb=1.0, ros=1.0, atp=2.0)
        assert float(ddis.senescence) > float(ctrl.senescence)
        assert float(ddis.autophagy) > float(ctrl.autophagy)
        assert float(ddis.oxphos) < float(ctrl.oxphos)


# ═══════════════════════════════════════════════════════════════════════════
# Batching / shape polymorphism
# ═══════════════════════════════════════════════════════════════════════════


class TestBatching:

    def test_batched_inputs_produce_batched_outputs(self):
        mapper = PathwayMapper()
        batch = jnp.linspace(0.1, 2.0, 16)
        scores = mapper.score(
            p53=batch, mtor=batch, nfkb=batch, ros=batch, atp=batch * 2
        )
        for v in scores.as_dict().values():
            assert v.shape == (16,)

    def test_2d_inputs_produce_2d_outputs(self):
        """Trajectory shape (n_time, batch) preserved through scoring."""
        mapper = PathwayMapper()
        x = jnp.ones((10, 4))  # (n_time, batch)
        scores = mapper.score(p53=x, mtor=x, nfkb=x, ros=x, atp=x * 2)
        for v in scores.as_dict().values():
            assert v.shape == (10, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Differentiability
# ═══════════════════════════════════════════════════════════════════════════


class TestDifferentiability:
    """The whole point of being an eqx.Module instead of post-hoc Python:
    gradients flow through pathway scores into mechanistic state and into
    Hill parameters."""

    def test_grad_senescence_wrt_p53_positive(self):
        mapper = PathwayMapper(K=0.5, n=4.0, K_atp=3.0)

        def loss(p53):
            return mapper.score(
                p53=p53,
                mtor=jnp.asarray(0.5),
                nfkb=jnp.asarray(0.5),
                ros=jnp.asarray(0.5),
                atp=jnp.asarray(1.0),
            ).senescence

        g = jax.grad(loss)(jnp.asarray(0.5))
        assert float(g) > 0.0

    def test_grad_oxphos_wrt_atp_positive(self):
        mapper = PathwayMapper(K=0.5, n=4.0, K_atp=3.0)

        def loss(atp):
            return mapper.score(
                p53=jnp.asarray(0.3),
                mtor=jnp.asarray(0.3),
                nfkb=jnp.asarray(0.3),
                ros=jnp.asarray(0.3),
                atp=atp,
            ).oxphos

        g = jax.grad(loss)(jnp.asarray(2.0))
        assert float(g) > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Correlation metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestCorrelationMetrics:

    def test_pearson_r_perfect_positive(self):
        a = jnp.linspace(0.0, 1.0, 7)
        assert jnp.isclose(pearson_r(a, a), 1.0, atol=1e-5)

    def test_pearson_r_perfect_negative(self):
        a = jnp.linspace(0.0, 1.0, 7)
        assert jnp.isclose(pearson_r(a, -a), -1.0, atol=1e-5)

    def test_pearson_r_uncorrelated(self):
        a = jnp.array([1.0, -1.0, 1.0, -1.0])
        b = jnp.array([1.0, 1.0, -1.0, -1.0])
        assert abs(float(pearson_r(a, b))) < 1e-5

    def test_sign_agreement_all_match(self):
        a = jnp.array([1.0, -1.0, 1.0, -1.0])
        assert float(sign_agreement(a, a)) == 1.0

    def test_sign_agreement_half(self):
        a = jnp.array([1.0, -1.0, 1.0, -1.0])
        b = jnp.array([1.0, 1.0, 1.0, 1.0])
        assert float(sign_agreement(a, b)) == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════════


class TestCalibration:
    """Synthetic ground-truth: generate Δ_data from a known mapper, then
    calibrate a different (default) mapper against it. r should improve
    substantially toward 1."""

    def test_calibration_improves_pearson(self):
        true_mapper = PathwayMapper(K=0.3, n=6.0, K_atp=2.0)
        ctrl = dict(p53=0.2, mtor=0.5, nfkb=0.3, ros=0.2, atp=4.0)
        pert = dict(p53=1.5, mtor=0.4, nfkb=1.0, ros=0.8, atp=2.5)
        true_delta = true_mapper.score(
            **{k: jnp.asarray(v) for k, v in pert.items()}
        ).as_vector(PATHWAY_ORDER) - true_mapper.score(
            **{k: jnp.asarray(v) for k, v in ctrl.items()}
        ).as_vector(
            PATHWAY_ORDER
        )

        initial = PathwayMapper()  # K=0.5, n=4.0 — different from true
        r_initial = float(
            pearson_r(
                initial.score(
                    **{k: jnp.asarray(v) for k, v in pert.items()}
                ).as_vector(PATHWAY_ORDER)
                - initial.score(
                    **{k: jnp.asarray(v) for k, v in ctrl.items()}
                ).as_vector(PATHWAY_ORDER),
                true_delta,
            )
        )

        inputs = {f"{k}_ctrl": jnp.asarray(v) for k, v in ctrl.items()}
        inputs.update({f"{k}_pert": jnp.asarray(v) for k, v in pert.items()})
        _, history = calibrate_pathway_mapper(
            delta_sim_inputs=inputs,
            delta_data=true_delta,
            initial=initial,
            n_steps=300,
            learning_rate=0.05,
        )

        r_final = float(history["r"][-1])
        # r should improve substantially. We don't require r=1 exactly
        # because we're only allowed to tune Hill K/n + senescence
        # weights, not the formula structure.
        assert r_final > r_initial
        assert r_final > 0.9


# ═══════════════════════════════════════════════════════════════════════════
# from_eriq_state — convenience binding to ERiQ store layout
# ═══════════════════════════════════════════════════════════════════════════


class TestFromEriqState:
    """``from_eriq_state`` is the ergonomic call that pulls ERiQ state
    paths and computes algebraic NF-κB / ATP under the hood."""

    def test_from_eriq_state_runs_and_returns_seven_pathways(self):
        mapper = PathwayMapper()
        # Stub state with all the required eriq/* paths; use the ERiQ
        # homeostatic IC values as a sanity-check input.
        state = {
            "eriq/mito_function": jnp.asarray(3.6239),
            "eriq/glycolysis": jnp.asarray(2.4010),
            "eriq/mito_damage": jnp.asarray(0.0724),
            "eriq/mTOR_activity": jnp.asarray(-0.1936),
            "eriq/p53_activity": jnp.asarray(0.8734),
            "eriq/ROS_activity": jnp.asarray(0.0794),
            "eriq/ROS_integrator_c": jnp.asarray(-0.7944),
        }
        scores = mapper.from_eriq_state(state)
        d = scores.as_dict()
        assert len(d) == 7
        for v in d.values():
            v = float(v)
            assert 0.0 <= v <= 1.0
