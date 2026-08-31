"""Tests for the Hill coupling edges: port contract, gate arithmetic, gate
enumeration, and placing a gate so its signal crosses a downstream critical
value. Every case runs on a toy composite — what a *demo* happens to be wired
to is not a property of the framework."""

import jax
import jax.numpy as jnp
import pytest

from hallsim.models.hill_edge import HillActivationEdge
from hallsim.process import PortRole


def _edge(**kw):
    kw.setdefault("k_act", 0.02)
    kw.setdefault("K", (4.0,))
    kw.setdefault("n", (2.0,))
    return HillActivationEdge(**kw)


class TestHillActivationEdge:
    def test_ports_are_generic(self):
        schema = _edge().ports_schema()
        assert schema["target"].role == PortRole.EVOLVED
        assert schema["target"].reads_value is False  # pure source
        assert schema["source"].role == PortRole.INPUT

    def test_only_target_in_derivative(self):
        dy = _edge().derivative(0.0, {"source": jnp.array(4.0)})
        assert set(dy.keys()) == {"target"}

    def test_activating_monotonic_and_saturating(self):
        e = _edge()
        low = float(e.derivative(0.0, {"source": jnp.array(2.8)})["target"])
        high = float(e.derivative(0.0, {"source": jnp.array(5.7)})["target"])
        assert 0.0 < low < high
        sat = float(e.derivative(0.0, {"source": jnp.array(1e4)})["target"])
        assert sat <= e.k_act + 1e-9  # Hill saturates at 1

    def test_half_saturation_at_K(self):
        # source == K -> drive == 0.5 -> target == k_act/2.
        d = _edge().derivative(0.0, {"source": jnp.array(4.0)})["target"]
        assert float(d) == pytest.approx(0.01, abs=1e-6)

    def test_differentiable_through_source_and_rate(self):
        g = jax.grad(
            lambda m: _edge().derivative(0.0, {"source": m})["target"]
        )(jnp.array(4.0))
        assert jnp.isfinite(g) and g > 0
        gk = jax.grad(
            lambda k: _edge(k_act=k).derivative(
                0.0, {"source": jnp.array(4.0)}
            )["target"]
        )(jnp.array(0.02))
        assert jnp.isfinite(gk) and gk > 0

    def test_k_act_is_the_calibratable_surface(self):
        assert {p.field for p in _edge().calibratable_params()} == {"k_act"}

    def test_declarative_metadata_folds_in(self):
        m = _edge(hallmark="H", reference="R", description="D").metadata()
        assert (m["hallmark"], m["reference"], m["description"]) == (
            "H",
            "R",
            "D",
        )

    def test_multi_source_gates_multiply(self):
        e = _edge(k_act=1.0, K=(1.0, 1.0), n=(2.0, 2.0), sources=("a", "b"))
        assert set(e.ports_schema()) == {"target", "a", "b"}
        d = e.derivative(0.0, {"a": jnp.array(1.0), "b": jnp.array(1.0)})
        assert float(d["target"]) == pytest.approx(0.25, abs=1e-6)


@pytest.mark.demo
@pytest.mark.network
class TestGateEnumerationAndRange:
    """A gate placed outside its driver's realised range is dead or saturated,
    and both look like a weak coupling."""

    def _composite(self, K):
        from hallsim.composite import Composite
        from hallsim.models.hill_edge import HillActivationEdge
        from hallsim.process import Port, PortRole, Process

        class Source(Process):
            rate: float = 1.0

            def ports_schema(self):
                return {"s": Port(role=PortRole.EVOLVED, default=1.0)}

            def derivative(self, t, state):
                return {"s": self.rate}

        return Composite(
            processes={
                "src": Source(),
                "gate": HillActivationEdge(
                    k_act=0.1, K=(K,), n=(2.0,), target_default=0.0
                ),
            },
            topology={
                "src": {"s": "pool/s"},
                "gate": {"source": "pool/s", "target": "pool/t"},
            },
            validate=False,
            semantic_validation=False,
        )

    def test_enumerates_gates_without_a_solve(self):
        gates = self._composite(5.0).hill_gates()
        assert gates == {"gate": (["pool/s"], (5.0,))}

    def test_a_process_without_K_is_not_a_gate(self):
        comp = self._composite(5.0)
        assert "src" not in comp.hill_gates()


class TestCrossingPlacement:
    """Placing a gate so its signal crosses a downstream bifurcation, rather
    than so it forms a clean on/off switch."""

    def _place(self, **kw):
        from hallsim.models.hill_edge import place_hill_gate_for_crossing

        return place_hill_gate_for_crossing(**kw)

    def test_reproduces_the_psi_hopf_window(self):
        """GZ06's p53 Hopf at psi=0.685416, driven from basal 0.3 to hi 1.0
        across a control ceiling of 9.59 and a DDIS peak of 27.18."""
        s = self._place(
            off_level=9.59,
            on_level=27.18,
            basal=0.3,
            hi=1.0,
            critical=0.685416,
            n=2.0,
        )
        assert s.ok
        assert s.K == pytest.approx(14.586, rel=1e-3)
        assert s.window[0] == pytest.approx(8.664, rel=1e-3)
        assert s.window[1] == pytest.approx(24.556, rel=1e-3)

    def test_reproduces_the_alpha_x_hopf_window_on_an_inhibitory_edge(self):
        """The shipped edge: hi (0.0) *below* basal (0.6648), Hopf at 0.1662."""
        s = self._place(
            off_level=9.59,
            on_level=12.13,
            basal=0.6648,
            hi=0.0,
            critical=0.1662,
            n=2.0,
        )
        assert s.ok
        assert s.K == pytest.approx(6.221, rel=1e-3)
        assert s.window[0] == pytest.approx(5.537, rel=1e-3)
        assert s.window[1] == pytest.approx(7.003, rel=1e-3)

    def test_the_crossing_lands_where_the_signal_equals_critical(self):
        from hallsim.models.hill_edge import hill_gate

        basal, hi, crit, n = 0.6648, 0.0, 0.1662, 2.0
        s = self._place(
            off_level=9.59,
            on_level=12.13,
            basal=basal,
            hi=hi,
            critical=crit,
            n=n,
        )
        signal = basal + (hi - basal) * float(
            hill_gate(jnp.asarray(s.crossing), s.K, n)
        )
        assert signal == pytest.approx(crit, rel=1e-6)

    def test_succeeds_where_a_clean_switch_cannot(self):
        """r=1.26 needs n=19 for a 10/90 gate; a crossing needs only off<on."""
        from hallsim.models.hill_edge import place_hill_gate

        assert not place_hill_gate(9.59, 12.13).ok
        assert self._place(
            off_level=9.59,
            on_level=12.13,
            basal=0.6648,
            hi=0.0,
            critical=0.1662,
            n=2.0,
        ).ok

    def test_unreachable_critical_is_refused(self):
        s = self._place(
            off_level=1.0,
            on_level=27.18,
            basal=0.3,
            hi=1.0,
            critical=1.5,
            n=2.0,
        )
        assert not s.ok
        assert "never reaches it" in s.note

    def test_overlapping_ranges_are_refused(self):
        s = self._place(
            off_level=12.0,
            on_level=9.0,
            basal=0.3,
            hi=1.0,
            critical=0.685416,
            n=2.0,
        )
        assert not s.ok
        assert "overlap" in s.note
