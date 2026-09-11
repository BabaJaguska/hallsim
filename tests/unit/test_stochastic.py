"""Tests for reaction-level SSA execution."""

import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hallsim.sbml_import import process_from_sbml
from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler
from hallsim.stochastic import SSACompatibilityError, simulate_ssa

MODEL = """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="decay">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="10"/>
      <species id="B" compartment="c" initialAmount="0"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="1"/></listOfParameters>
    <listOfReactions>
      <reaction id="decay">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_ssa_uses_source_stoichiometry_and_preserves_counts(tmp_path):
    path = tmp_path / "decay.xml"
    path.write_text(textwrap.dedent(MODEL))
    process = process_from_sbml(str(path), name="decay")

    result = simulate_ssa(
        process,
        t_span=(0.0, 3.0),
        save_dt=0.5,
        seed=4,
    )

    assert result.states.shape == (7, 2)
    assert result.reaction_indices.size > 0
    assert np.all(result.states >= 0)
    assert np.all(result.states[:, 0] + result.states[:, 1] == 10)
    channel = process.reaction_channels()[0]
    assert dict(channel.stoichiometry) == {"A": -1.0, "B": 1.0}


def test_ssa_rejects_negative_propensities():
    class Negative:
        _species_names = ("A",)
        _species_y0 = (1.0,)

        def reaction_channels(self):
            from hallsim.sbml_import import SBMLReactionChannel

            return (
                SBMLReactionChannel(
                    reaction_id="bad",
                    rate_law="-1",
                    stoichiometry=(("A", 1.0),),
                ),
            )

        def reaction_propensities(self, t, state):
            return np.asarray([-1.0])

    import pytest

    with pytest.raises(ValueError, match="negative"):
        simulate_ssa(Negative(), t_span=(0.0, 1.0))


def test_ssa_rejects_fractional_initial_counts():
    class Fractional:
        _species_names = ("A",)
        _species_y0 = (0.5,)

        def reaction_channels(self):
            from hallsim.sbml_import import SBMLReactionChannel

            return (
                SBMLReactionChannel(
                    reaction_id="bad",
                    rate_law="1",
                    stoichiometry=(("A", 1.0),),
                ),
            )

        def reaction_propensities(self, t, state):
            return np.asarray([1.0])

    import pytest

    with pytest.raises(SSACompatibilityError, match="integer"):
        simulate_ssa(Fractional(), t_span=(0.0, 1.0))


def test_scheduler_can_advance_an_explicit_stochastic_process(tmp_path):
    path = tmp_path / "decay.xml"
    path.write_text(textwrap.dedent(MODEL))
    process = process_from_sbml(str(path), name="decay").as_stochastic()
    composite = Composite(
        processes={"decay": process},
        topology={
            "decay": {name: f"decay/{name}" for name in process._species_names}
        },
        validate=False,
        semantic_validation=False,
    )
    result = Scheduler().run(
        composite,
        t_span=(0.0, 1.0),
        macro_dt=0.25,
        save_dt=0.25,
        seed=7,
    )
    assert result.stats["decay"]["num_events"] > 0
    assert np.all(result.ys[:, 0] + result.ys[:, 1] == 10)


def test_composite_rejects_continuous_writer_on_stochastic_count(tmp_path):
    path = tmp_path / "decay.xml"
    path.write_text(textwrap.dedent(MODEL))
    stochastic = process_from_sbml(str(path), name="decay").as_stochastic()

    class ContinuousWriter(Process):
        def ports_schema(self):
            return {"A": Port(role=PortRole.EVOLVED, default=0.0)}

        def derivative(self, t, state):
            return {"A": 0.25}

    import pytest

    with pytest.raises(ValueError, match="Stochastic count path"):
        Composite(
            processes={"decay": stochastic, "writer": ContinuousWriter()},
            topology={
                "decay": {
                    name: f"cell/{name}" for name in stochastic._species_names
                },
                "writer": {"A": "cell/A"},
            },
            semantic_validation=False,
        )


def _decay_composite(tmp_path, initial_amount=10):
    path = tmp_path / "decay.xml"
    path.write_text(
        textwrap.dedent(MODEL).replace(
            'initialAmount="10"', f'initialAmount="{initial_amount}"'
        )
    )
    process = process_from_sbml(str(path), name="decay").as_stochastic()
    return Composite(
        processes={"decay": process},
        topology={
            "decay": {name: f"decay/{name}" for name in process._species_names}
        },
        validate=False,
        semantic_validation=False,
    )


_SPAN = dict(t_span=(0.0, 1.0), macro_dt=0.25, save_dt=0.25)


def test_batched_stochastic_members_draw_independent_noise(tmp_path):
    composite = _decay_composite(tmp_path, initial_amount=200)
    y0 = jnp.stack([composite.initial_state_vec()] * 3)

    def run(seed):
        return Scheduler().run(composite, y0=y0, seed=seed, **_SPAN)

    result = run(3)
    ys = np.asarray(result.ys)
    assert ys.shape == (5, 3, 2)
    assert np.all(ys[..., 0] + ys[..., 1] == 200)
    same_01 = np.array_equal(ys[:, 0], ys[:, 1])
    same_12 = np.array_equal(ys[:, 1], ys[:, 2])
    assert not (same_01 and same_12), ys[-1]
    assert np.all(np.asarray(result.stats["decay"]["num_events"]) > 0)
    assert np.array_equal(np.asarray(run(3).ys), ys)
    assert not np.array_equal(np.asarray(run(4).ys), ys)


def test_key_argument_matches_seed(tmp_path):
    composite = _decay_composite(tmp_path)
    by_seed = Scheduler().run(composite, seed=7, **_SPAN)
    by_key = Scheduler().run(composite, key=jax.random.PRNGKey(7), **_SPAN)
    assert np.array_equal(np.asarray(by_seed.ys), np.asarray(by_key.ys))


def test_eager_lane_is_seeded_and_refuses_a_batch(tmp_path):
    composite = _decay_composite(tmp_path)
    eager = Scheduler(progress=True)
    first = eager.run(composite, seed=7, **_SPAN)
    again = eager.run(composite, seed=7, **_SPAN)
    assert np.array_equal(np.asarray(first.ys), np.asarray(again.ys))
    assert np.all(first.ys[:, 0] + first.ys[:, 1] == 10)
    y0 = jnp.stack([composite.initial_state_vec()] * 2)
    with pytest.raises(ValueError, match="unbatched"):
        eager.run(composite, y0=y0, **_SPAN)
