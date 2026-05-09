"""P53Bridge — drive ERiQ's p53_activity from an external p53 model.

The bridge is a thin tracking-dynamic Process that lets a HallSim composite
*replace* ERiQ's intrinsic algebraic p53 with the dynamic output of an
external SBML model (e.g. Geva-Zatorsky 2006 BIOMD0000000157 — see
:mod:`hallsim.models.damage_repair` for the upstream half of the story).

Why this needs to be a Process at all
-------------------------------------
Topology in HallSim wires named ports to store paths but does not apply
algebraic transforms. To rescale or shift an external state variable
into ERiQ's expected range, the transform has to live inside a Process's
``derivative``. The Bridge owns ERiQ's ``p53_activity`` store path
(EXCLUSIVE) and writes a fast-tracking dynamic of the form
``dy/dt = k_track * (gain * source + offset - y)``, so that y converges
to ``gain * source + offset`` on a timescale of ``1/k_track`` regardless
of source dynamics (constant, oscillatory, ramping).

When the bridge is in a composite, ERiQ should use the
:class:`hallsim.models.eriq.ERiQSignalingNoP53` variant — it omits the
EXCLUSIVE p53 port so the Bridge can own it without a topology conflict.

Parameters
----------
source_port:
    Port name on which to read the source p53 value. Default ``"p53_x"``,
    matching the convention of routing GZ06's ``x`` to a store path
    named ``p53/x`` (and the bridge's source_port is wired to that path).
gain, offset:
    Linear rescaling: target = gain * source + offset.
k_track:
    Tracking rate constant (1/time). Larger = faster response. Default
    5.0 → ~0.2 time-unit lag.
"""

from __future__ import annotations

from hallsim.process import Port, PortRole, Process


class P53Bridge(Process):
    """Track ERiQ's p53_activity to a rescaled external p53 source.

    EXCLUSIVE owner of ``p53_activity``. Reads ``p53_x`` (or whatever
    port name is configured) as INPUT — this is wired in topology to
    the store path of the external p53 model's protein state variable.
    """

    gain: float = 1.0
    offset: float = 0.0
    k_track: float = 5.0

    def ports_schema(self):
        return {
            "p53_x": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=(
                    "External p53 protein level (e.g. Geva-Zatorsky x). "
                    "Wired in topology to the SBML p53 model's state path."
                ),
                ontology={"go": "GO:0030330"},  # DNA damage response, p53
            ),
            "p53_activity": Port(
                role=PortRole.EXCLUSIVE,
                default=0.8734,
                units="dimensionless",
                description=(
                    "ERiQ p53_activity (Ax). Owned by the bridge — the "
                    "ERiQSignalingNoP53 variant must be used in the same "
                    "composite to avoid double-ownership conflict."
                ),
                ontology={"go": "GO:0030330"},
            ),
        }

    def derivative(self, t, state):
        target = self.gain * state["p53_x"] + self.offset
        return {
            "p53_activity": self.k_track * (target - state["p53_activity"]),
        }

    def metadata(self):
        base = super().metadata()
        base["description"] = (
            "Replaces ERiQ's intrinsic p53_activity dynamics with a tracking "
            "dynamic to an external (e.g. SBML-imported) p53 source. Use "
            "alongside ERiQSignalingNoP53 to avoid topology conflict."
        )
        return base
