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

from hallsim.calibration import (
    CalibrationProblem,
    Calibrator,
    Condition,
    ParameterRef,
)
from hallsim.composite import Composite
from hallsim.hallmarks import (
    HALLMARK_REGISTRY,
    HallmarkHandle,
    ParameterMapping,
    apply_hallmarks,
)
from hallsim.process import Port, PortRole, Process, ProcessKind
from hallsim.scheduler import EventRecord, Scheduler, SchedulerResult
from hallsim.validation import (
    ComposabilityReport,
    CompositeValidator,
    OverlapMatch,
    analyze_composability,
)

__all__ = [
    "CalibrationProblem",
    "Calibrator",
    "ComposabilityReport",
    "Composite",
    "CompositeValidator",
    "Condition",
    "EventRecord",
    "HALLMARK_REGISTRY",
    "HallmarkHandle",
    "OverlapMatch",
    "ParameterMapping",
    "ParameterRef",
    "Port",
    "PortRole",
    "Process",
    "ProcessKind",
    "Scheduler",
    "SchedulerResult",
    "analyze_composability",
    "apply_hallmarks",
]
