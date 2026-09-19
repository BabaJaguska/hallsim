"""A process without a timescale is grouped with what it couples."""

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
import equinox as eqx


class Relax(Process):
    """dx/dt = (u - x) / tau, with a declared timescale."""

    tau: float = 1.0
    timescale: float | None = eqx.field(static=True, default=1.0)

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=0.0),
            "u": Port(role=PortRole.INPUT, default=0.0),
        }

    def derivative(self, t, state):
        return {"x": (state["u"] - state["x"]) / self.tau}


class Edge(Process):
    """Reads a source, writes a target; declares no timescale."""

    timescale: float | None = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            "src": Port(role=PortRole.INPUT, default=0.0),
            "out": Port(role=PortRole.EXCLUSIVE, default=0.0),
        }

    def derivative(self, t, state):
        return {"out": state["src"] - state["out"]}


def _groups(topology, procs):
    comp = Composite(
        processes=procs,
        topology=topology,
        validate=False,
        semantic_validation=False,
    )
    return comp.auto_groups()


def test_an_untimed_edge_joins_the_group_of_the_model_it_drives():
    procs = {
        "fast": Relax(tau=1.0, timescale=1.0),
        "slow": Relax(tau=1e4, timescale=1e4),
        "edge": Edge(),
    }
    topo = {
        "fast": {"x": "xf", "u": "drive"},
        "slow": {"x": "xs", "u": "coupling"},
        "edge": {"src": "xf", "out": "coupling"},
    }
    groups = _groups(topo, procs)
    home = next(g for g, ps in groups.items() if "edge" in ps)
    assert "slow" in groups[home]
    assert "default" not in groups


def test_an_untimed_process_coupled_to_nothing_stands_alone():
    procs = {"fast": Relax(), "loner": Edge()}
    topo = {
        "fast": {"x": "xf", "u": "drive"},
        "loner": {"src": "nothing", "out": "nowhere"},
    }
    groups = _groups(topo, procs)
    assert groups.get("default") == ["loner"]
