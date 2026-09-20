"""ImportedODEProcess — shared base for models imported from an ODE format.

Holds what every importer (SBML, XPP) needs: the native-time clock and its
chain-rule reconciliation, and the ``parameters`` dict as the fittable
calibration surface — so a change to either lands once, for both.

Subclasses supply the format-specific parts (parsed model, ports,
``derivative``, ``coupling_structure``) and set ``_param_label``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process


class ParamInput(eqx.Module):
    """Exposes an imported model's constant as a plain INPUT port, taking the
    port's value directly each step.

    The transform-free primitive for parameter coupling: put any Hill / gate /
    product in a composable edge that writes the driving path (e.g.
    :class:`hallsim.models.hill_edge.HillEdge` in ``level`` mode), then this reads it.
    """

    param_name: str = eqx.field(static=True)
    input_port: str = eqx.field(static=True)

    def value(self, basal, signal):
        return jnp.asarray(signal)

    def symbolic(self, basal, signal):
        """:meth:`value` as sympy, for the owner's symbolic forms."""
        return signal


def _scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ImportedODEProcess(Process):
    """Base for an ODE model auto-generated from an external format.

    Not constructed directly — a format importer (``process_from_sbml`` /
    ``process_from_xpp``) builds the concrete subclass.
    """

    native_time_seconds: float = 1.0
    # Where native_time_seconds came from: ``"declared"`` (the source asserts a
    # time unit), ``"supplied"`` (the caller passed the true value the source
    # omits), or ``"assumed"`` (a fallback guess — SBML's default is seconds).
    # Only ``"assumed"`` is unverified, and reconciling an assumed clock onto a
    # shared axis is silently 60×/3600×/86400× wrong. Hand-built processes
    # declare their own.
    native_time_source: str = eqx.field(static=True, default="declared")
    # Native time per composite time unit: a unit conversion, so structure.
    time_scale: float = eqx.field(static=True, default=1.0)
    # The calibration surface — traced, so Calibrator/hallmarks differentiate
    # through it. Everything below is *structure*: names, index maps, port
    # defaults. Static, so ports_schema() stays concrete under a trace and
    # Scheduler.run can be jitted end to end.
    parameters: dict[str, float] = None  # type: ignore[assignment]
    _param_names: tuple[str, ...] = eqx.field(static=True, default=())
    _name: str = eqx.field(static=True, default="")
    # Each drives one imported constant from an INPUT port every step (see
    # ParamInput). Static, so it round-trips untouched through the tree_at
    # substitutions hallmarks and Calibrator apply to `parameters`.
    _param_drivers: tuple = eqx.field(static=True, default=())
    # Provenance: the accession or path asked for, the file actually read,
    # its SHA-256, and the deposit's own parameter values before any
    # override — so a result can say what it was computed from.
    source: str = eqx.field(static=True, default="")
    source_path: str = eqx.field(static=True, default="")
    source_sha256: str = eqx.field(static=True, default="")
    _published_parameters: tuple = eqx.field(static=True, default=())

    _param_label = "parameter"  # "SBML constant" / "XPP parameter"

    def with_param_input(self, param_name: str, input_port: str):
        """Copy exposing constant ``param_name`` as an INPUT port; wire it to a
        driving store path via topology. See :class:`ParamInput`."""
        return self._add_param_driver(
            ParamInput(
                param_name=self._check_param(param_name), input_port=input_port
            )
        )

    def without_events(self, *names: str):
        """Copy carrying none of its events, or none of the ``names`` given,
        so composing it discards them.

        A Composite expands a member process's events automatically, because
        forgetting to do so ran models with their input route removed while
        returning smooth, bounded numbers. Discarding them is
        legitimate but deliberate — it is what replacing a model's own
        event-delivered insult with an external ``u(t)`` driver requires, per
        acceptance test 5 in ``docs/senescence-model-rebuild.md``. Written as a
        method so the discard is visible where it is decided::

            Composite(processes={"dp14": dp14.without_events()}, ...)

        A name is the SBML event id or the translated process name.
        """
        import copy

        events = tuple(getattr(self, "_events", ()))
        keep = ()
        if names:

            def matches(ev, n):
                return ev._name in (n, f"{self._name}__{n}")

            missing = [
                n for n in names if not any(matches(ev, n) for ev in events)
            ]
            if missing:
                raise KeyError(
                    f"{missing} are not events on {self._name!r}; "
                    f"available: {[ev._name for ev in events]}"
                )
            keep = tuple(
                ev for ev in events if not any(matches(ev, n) for n in names)
            )
        new = copy.copy(self)
        object.__setattr__(new, "_events", keep)
        return new

    def protocol(self) -> list[dict]:
        """The model's own timed inputs: every event it carries, with its
        trigger time in native units and what it sets
        (:func:`hallsim.sbml_events.event_schedule`)."""
        from hallsim.sbml_events import event_schedule

        return event_schedule(getattr(self, "_events", ()))

    def _check_param(self, param_name: str) -> str:
        if param_name not in self._param_names:
            raise KeyError(
                f"{param_name!r} is not a constant on {self._name!r}; "
                f"available: {sorted(self._param_names)}"
            )
        return param_name

    def _add_param_driver(self, driver):
        # Drivers are pure static metadata (no array leaves), so tree_at can't
        # grow the tuple; copy + set the field directly.
        import copy

        new = copy.copy(self)
        object.__setattr__(
            new, "_param_drivers", self._param_drivers + (driver,)
        )
        return new

    def _driver_input_ports(self) -> dict:
        """INPUT ports feeding the live parameter drivers, to be merged into
        the subclass ``ports_schema``. Wire each to its driving store path via
        topology."""
        return {
            d.input_port: Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description=f"drives {self._param_label} {d.param_name!r}",
            )
            for d in self._param_drivers
        }

    def _driven_param_values(self, state) -> dict:
        """``{param_name: value}`` per live driver, which each format's
        ``derivative`` writes onto its own constant representation."""
        return {
            d.param_name: d.value(
                self.parameters[d.param_name], state[d.input_port]
            )
            for d in self._param_drivers
        }

    def reconciled_to(self, canonical_time_seconds: float):
        """Copy on the composite's canonical clock, chain-rule rescaling the
        native rate law by ``canonical_time_seconds / native_time_seconds``.
        ``canonical_time_seconds`` is the real-world duration of one ``t_span``
        unit (86400 for a day axis); Scheduler grouping is separate."""
        from hallsim.process import write_param

        scale = canonical_time_seconds / float(self.native_time_seconds)
        return write_param(self, "time_scale", float(scale))

    def frozen_species(self) -> list[str]:
        """Species import holds at their initial value; none unless the
        importer freezes inert sinks."""
        return []

    def provenance(self) -> dict:
        """Where this model came from and what import did to it: the source
        asked for, the file read and its SHA-256, the native clock and its
        reconciliation factor, the species frozen at import, and every
        parameter whose value differs from the deposit's."""
        published = dict(self._published_parameters)
        current = {k: _scalar(v) for k, v in (self.parameters or {}).items()}
        return {
            "source": self.source,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "native_time_seconds": self.native_time_seconds,
            "native_time_source": self.native_time_source,
            "time_scale": _scalar(self.time_scale),
            "events": self.protocol(),
            "frozen_species": self.frozen_species(),
            "published_parameters": published,
            "modified_parameters": {
                k: v for k, v in current.items() if published.get(k) != v
            },
        }

    def metadata(self):
        base = super().metadata()
        base.update(self.provenance())
        base["n_parameters"] = len(self._param_names)
        return base

    def calibratable_params(self) -> list:
        """Every imported parameter as a fittable ``parameters.<name>``, plus
        any :func:`~hallsim.process.calibratable` field on the subclass.
        Exposing all of them is safe —
        :meth:`Composite.calibration_targets` filters hallmark-controlled
        knobs."""
        from hallsim.calibration import CalibratableParam

        out = super().calibratable_params()
        skip = getattr(self, "_compartment_names", frozenset())
        for name, value in self.parameters.items():
            if name in skip:
                continue
            v = float(value)
            out.append(
                CalibratableParam(
                    process_name="",
                    field=f"parameters.{name}",
                    default=v,
                    clamp=None,
                    description=f"{self._param_label} {name!r} on {self._name}",
                )
            )
        return out
