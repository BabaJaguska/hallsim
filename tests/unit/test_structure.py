"""Composite-wide structure (hallsim.structure): stoichiometry over paths,
the Jacobian's sparsity and colouring, and the field as sympy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy

from hallsim.composite import Composite
from hallsim.models.clamp_edge import ClampEdge
from hallsim.models.gain_edge import GainEdge
from hallsim.models.running_integral import RunningIntegral
from hallsim.process import Port, PortRole, Process, ReactionChannel
from hallsim.structure import (
    StructureError,
    check_pattern,
    composite_moieties,
    composite_stoichiometry,
    compressed_jacobian,
    jacobian_pattern,
    rational_null_space,
    symbolic_field,
)


def _ring(n, tau=2.5):
    """``n`` leaky integrals in a ring, ``dx_i/dt = x_{i-1} − x_i/τ``:
    declared, banded, no conserved quantity."""
    paths = [f"net/x{i}" for i in range(n)]
    return Composite(
        processes={
            f"p{i}": RunningIntegral(power=1.0, tau=tau, initial=1.0)
            for i in range(n)
        },
        topology={
            f"p{i}": {"integral": paths[i], "source": paths[(i - 1) % n]}
            for i in range(n)
        },
        semantic_validation=False,
    )


def _rhs(comp):
    rhs, keys = comp.build_rhs()
    return (lambda y: rhs(0.0, y)), keys


class Bind(Process):
    """E + S ⇌ ES through one declared channel."""

    kon: float = 1.0
    koff: float = 0.5

    def ports_schema(self):
        return {
            "E": Port(role=PortRole.EVOLVED, default=1.0),
            "S": Port(role=PortRole.EVOLVED, default=2.0),
            "ES": Port(role=PortRole.EVOLVED, default=0.0),
        }

    def derivative(self, t, state):
        v = self.kon * state["E"] * state["S"] - self.koff * state["ES"]
        return {"E": -v, "S": -v, "ES": v}

    def reaction_channels(self):
        E, S, ES, kon, koff = sympy.symbols("E S ES kon koff")
        return (
            ReactionChannel(
                "bind",
                (("E", -1.0), ("S", -1.0), ("ES", 1.0)),
                kon * E * S - koff * ES,
            ),
        )


class Opaque(Process):
    """Two coupled states and no symbolic form."""

    def ports_schema(self):
        return {
            "a": Port(role=PortRole.EVOLVED, default=1.0),
            "b": Port(role=PortRole.EVOLVED, default=1.0),
        }

    def derivative(self, t, state):
        return {"a": -state["a"] * state["b"], "b": -state["b"]}


def test_rational_null_space_is_the_null_space():
    rng = np.random.default_rng(0)
    a = rng.integers(-2, 3, size=(4, 7))
    basis = rational_null_space(a.tolist(), 7)
    m = sympy.Matrix(a.tolist())
    assert len(basis) == 7 - m.rank()
    for vec in basis:
        x = sympy.Matrix([vec.get(j, 0) for j in range(7)])
        assert all(v == 0 for v in m * x)


def test_moieties_are_exact_over_paths_and_ignore_the_undeclared():
    comp = Composite(
        processes={"b": Bind(), "o": Opaque()},
        topology={
            "b": {"E": "p/E", "S": "p/S", "ES": "p/ES"},
            "o": {"a": "p/a", "b": "p/b"},
        },
        semantic_validation=False,
    )
    structure = composite_stoichiometry(comp)
    keys = comp.store_keys()
    assert structure.opaque == ("o",)
    assert {keys[i] for i in structure.undescribed} == {"p/a", "p/b"}
    assert {keys[i] for i in structure.described} == {"p/E", "p/ES", "p/S"}
    # the free-column basis of the exact left null space of N
    assert composite_moieties(comp) == [
        {"p/E": 1, "p/ES": 1},
        {"p/E": 1, "p/S": -1},
    ]


def test_ring_pattern_is_banded_and_coloured_in_three():
    pattern = jacobian_pattern(_ring(12))
    assert pattern.nnz == 24  # self and predecessor per row
    assert pattern.n_colours <= 3


def test_compressed_jacobian_equals_jacfwd():
    comp = _ring(9)
    fn, keys = _rhs(comp)
    y = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, len(keys)))
    pattern = jacobian_pattern(comp)
    dense = np.asarray(jax.jacfwd(fn)(y))
    sparse = np.asarray(compressed_jacobian(fn, y, pattern))
    assert np.allclose(sparse, dense, atol=1e-12)
    check_pattern(fn, y, pattern, keys)


def test_undeclared_block_is_dense_only_on_itself():
    comp = Composite(
        processes={
            "o": Opaque(),
            "r": RunningIntegral(power=1.0, tau=1.0),
        },
        topology={
            "o": {"a": "p/a", "b": "p/b"},
            "r": {"integral": "p/i", "source": "p/a"},
        },
        semantic_validation=False,
    )
    keys = comp.store_keys()
    pattern = jacobian_pattern(comp)
    pairs = {(keys[r], keys[c]) for r, c in zip(pattern.rows, pattern.cols)}
    assert pairs == {
        ("p/a", "p/a"),
        ("p/a", "p/b"),
        ("p/b", "p/a"),
        ("p/b", "p/b"),
        ("p/i", "p/i"),
        ("p/i", "p/a"),
    }
    fn, _ = _rhs(comp)
    y = comp.initial_state_vec()
    assert np.allclose(compressed_jacobian(fn, y, pattern), jax.jacfwd(fn)(y))


def test_an_assigned_path_is_read_through_its_rule():
    """A flux edge fed by a level edge depends on what the level reads, and
    not on the assigned slot, which the RHS overwrites before reading."""
    comp = Composite(
        processes={
            "lvl": GainEdge(offset=0.0, gain=2.0, mode="level"),
            "flux": GainEdge(
                offset=0.0, gain=1.0, mode="flux", target_default=0.5
            ),
            "hold": ClampEdge(k_clamp=1.0),
        },
        topology={
            "lvl": {"source": "p/x", "signal": "lvl/level"},
            "flux": {"source": "lvl/level", "target": "p/y"},
            "hold": {"target": "p/x", "setpoint": "hold/sp"},
        },
        semantic_validation=False,
    )
    keys = comp.store_keys()
    pattern = jacobian_pattern(comp)
    pairs = {(keys[r], keys[c]) for r, c in zip(pattern.rows, pattern.cols)}
    assert ("p/y", "p/x") in pairs
    assert ("p/y", "lvl/level") not in pairs
    fn, _ = _rhs(comp)
    y = comp.initial_state_vec()
    check_pattern(fn, y, pattern, keys)
    assert np.allclose(compressed_jacobian(fn, y, pattern), jax.jacfwd(fn)(y))


def test_a_declaration_that_omits_a_read_is_caught():
    class Liar(Process):
        def ports_schema(self):
            return {
                "x": Port(role=PortRole.EVOLVED, default=1.0),
                "u": Port(role=PortRole.INPUT, default=0.5),
            }

        def derivative(self, t, state):
            return {"x": -state["x"] + state["u"]}

        def reaction_channels(self):
            return (
                ReactionChannel("decay", (("x", -1.0),), sympy.Symbol("x")),
            )

    comp = Composite(
        processes={"a": Liar(), "b": Liar()},
        topology={
            "a": {"x": "p/x", "u": "p/y"},
            "b": {"x": "p/y", "u": "p/x"},
        },
        semantic_validation=False,
    )
    fn, keys = _rhs(comp)
    with pytest.raises(StructureError, match="p/x"):
        check_pattern(
            fn, comp.initial_state_vec(), jacobian_pattern(comp), keys
        )


def test_pattern_is_available_under_a_trace():
    """The pattern is structure, so a jitted steady state can build it from
    a composite whose parameters are tracers."""
    import equinox as eqx

    comp = _ring(4)
    params, static = eqx.partition(comp, eqx.is_inexact_array)

    @jax.jit
    def colours(p):
        return jacobian_pattern(eqx.combine(p, static)).n_colours

    assert int(colours(params)) == jacobian_pattern(comp).n_colours


def test_symbolic_field_names_paths_and_parameters():
    field = symbolic_field(_ring(3))
    x0, x2 = sympy.Symbol("net/x0"), sympy.Symbol("net/x2")
    tau, power = sympy.Symbol("p0.tau"), sympy.Symbol("p0.power")
    assert (
        sympy.simplify(field.derivatives["net/x0"] - (x2**power - x0 / tau))
        == 0
    )
    assert float(field.parameters["p0.tau"]) == 2.5
    assert field.opaque == ()
