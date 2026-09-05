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
    :class:`hallsim.models.hill_edge.HillSignalEdge`), then this reads it.
    """

    param_name: str = eqx.field(static=True)
    input_port: str = eqx.field(static=True)

    def value(self, basal, signal):
        return jnp.asarray(signal)


class ImportedODEProcess(Process):
    """Base for an ODE model auto-generated from an external format.

    Not constructed directly — a format importer (``process_from_sbml`` /
    ``process_from_xpp``) builds the concrete subclass.
    """

    native_time_seconds: float = 1.0
    # Did the source actually declare its time unit, or is native_time_seconds
    # a fallback guess (SBML default = seconds)? False means the clock is
    # unverified: reconciling / composing it onto a shared axis can be silently
    # 60×/3600×/86400× wrong. Set at import; True for hand-built processes.
    native_time_declared: bool = eqx.field(static=True, default=True)
    time_scale: float = 1.0
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

    _param_label = "parameter"  # "SBML constant" / "XPP parameter"

    def with_param_input(self, param_name: str, input_port: str):
        """Copy exposing constant ``param_name`` as an INPUT port; wire it to a
        driving store path via topology. See :class:`ParamInput`."""
        return self._add_param_driver(
            ParamInput(
                param_name=self._check_param(param_name), input_port=input_port
            )
        )

    def without_events(self):
        """Copy carrying no events, so composing it discards them.

        A Composite expands a member process's events automatically, because
        forgetting to do so ran models with their input route removed while
        returning smooth, bounded numbers (P0.36). Discarding them is
        legitimate but deliberate — it is what replacing a model's own
        event-delivered insult with an external ``u(t)`` driver requires, per
        acceptance test 5 in ``docs/senescence-model-rebuild.md``. Written as a
        method so the discard is visible where it is decided::

            Composite(processes={"dp14": dp14.without_events()}, ...)
        """
        import copy

        new = copy.copy(self)
        object.__setattr__(new, "_events", ())
        return new

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
        scale = canonical_time_seconds / self.native_time_seconds
        # jnp, not float: tree_at skips __check_init__, and a float leaf is
        # static — the clock ratio would recompile per factor.
        return eqx.tree_at(lambda p: p.time_scale, self, jnp.asarray(scale))

    def metadata(self):
        base = super().metadata()
        base["native_time_seconds"] = self.native_time_seconds
        base["native_time_declared"] = self.native_time_declared
        base["time_scale"] = self.time_scale
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
        for name, value in self.parameters.items():
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
