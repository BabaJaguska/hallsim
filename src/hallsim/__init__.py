"""HallSim — composable, multi-scale simulator for aging biology.

Public API. Most users only need a handful of names:

>>> from hallsim import Process, Port, PortRole, ProcessKind
>>> from hallsim import Composite, Scheduler

The ``Process`` / ``Port`` / ``PortRole`` triple is the building-block
contract: declare ports with roles, implement ``derivative`` (or
``update`` / ``condition`` + ``handler``). ``Composite`` wires processes
via topology; ``Scheduler`` is the runner.

Hallmark handles, validation, plotting, and SBML import are imported on
demand from their respective submodules — they aren't surfaced here to
keep the top-level namespace small.

Validation against transcriptomic data is via
:mod:`hallsim.gene_reporters` — a one-to-one mapping from mechanistic
state variables to canonical reporter genes, evaluated by sign agreement
and Spearman concordance.
"""

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process, ProcessKind
from hallsim.scheduler import EventRecord, Scheduler, SchedulerResult

__all__ = [
    "Composite",
    "EventRecord",
    "Port",
    "PortRole",
    "Process",
    "ProcessKind",
    "Scheduler",
    "SchedulerResult",
]
