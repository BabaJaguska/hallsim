"""ImportedODEProcess — shared base for models imported from an ODE format.

Carries the machinery common to every format importer (SBML, XPP): the
native-time clock and its chain-rule reconciliation, and the
``parameters`` dict as the fittable calibration surface. Keeping it here
means the importers stay in lockstep on time handling and parameter
discovery — a change to reconciliation or calibratable extraction lands
once, for both.

Subclasses supply the format-specific parts — the parsed model, ports,
``derivative``, and ``coupling_structure`` — and set ``_param_label`` to
name their parameters in calibration descriptions.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process


class ParamInput(eqx.Module):
    """Exposes an imported model's constant as a plain INPUT port.

    Each derivative step the parameter takes the port's value directly
    (identity) — an external process supplies it as a computed store-path
    value. The transform-free primitive for parameter coupling: put any Hill /
    gate / product in a composable edge (e.g.
    :class:`hallsim.models.hill_edge.HillSignalEdge`) that writes the driving
    path, then this reads the result.
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
    time_scale: float = 1.0
    parameters: dict[str, float] = None  # type: ignore[assignment]
    _param_names: tuple[str, ...] = ()
    _name: str = ""
    # Live parameter couplings: each drives one imported constant from an INPUT
    # port every derivative step (see :class:`ParamInput`). Static metadata so
    # it round-trips untouched through the ``eqx.tree_at`` substitutions the
    # hallmark / Calibrator paths apply to ``parameters``. The subclass
    # ``derivative`` applies them via :meth:`_driven_param_values`, each
    # bridging ``param_name`` to its own constant representation.
    _param_drivers: tuple = eqx.field(static=True, default=())

    # Label for this format's parameters in calibration descriptions,
    # e.g. "SBML constant" / "XPP parameter". Class attribute, not a field.
    _param_label = "parameter"

    def with_param_input(self, param_name: str, input_port: str):
        """Return a copy exposing constant ``param_name`` as a plain INPUT port
        ``input_port``: each step the parameter takes the port's value directly
        (wire the port to a driving store path via topology).

        The general parameter-coupling primitive — no transform baked in.
        Compose the transform (Hill, gate, product of several sources) as an
        edge that writes the driving path (e.g.
        :class:`hallsim.models.hill_edge.HillSignalEdge`), then this reads the
        result."""
        return self._add_param_driver(
            ParamInput(
                param_name=self._check_param(param_name), input_port=input_port
            )
        )

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
        """``{param_name: driven_value}`` for every live driver this step —
        each format's ``derivative`` writes these onto its own constant
        representation (SBML: the ``c`` vector by index; XPP: the eval
        namespace by name)."""
        return {
            d.param_name: d.value(
                self.parameters[d.param_name], state[d.input_port]
            )
            for d in self._param_drivers
        }

    def reconciled_to(self, canonical_time_seconds: float):
        """Return a copy on the composite's canonical clock.

        Sets ``time_scale = canonical_time_seconds / native_time_seconds``
        so the native-time rate law is chain-rule-rescaled onto the shared
        axis. ``canonical_time_seconds`` is the real-world duration of one
        ``t_span`` unit (e.g. ``86400.0`` for a day axis). Scheduler
        grouping is handled separately by ``timescale`` (set to
        ``native_time_seconds`` at import, canonical-independent).
        """
        scale = canonical_time_seconds / self.native_time_seconds
        return eqx.tree_at(lambda p: p.time_scale, self, float(scale))

    def metadata(self):
        base = super().metadata()
        base["native_time_seconds"] = self.native_time_seconds
        base["time_scale"] = self.time_scale
        base["n_parameters"] = len(self._param_names)
        return base

    def calibratable_params(self) -> list:
        """Every imported parameter as a fittable ``parameters.<name>``.

        One :class:`hallsim.calibration.CalibratableParam` per entry in
        :attr:`parameters` (current value as default, two-order clamp),
        composed with any :func:`hallsim.process.calibratable` field on the
        subclass. :meth:`hallsim.composite.Composite.calibration_targets`
        filters hallmark-controlled knobs, so exposing all of them is safe.
        """
        from hallsim.calibration import CalibratableParam, default_clamp

        out = super().calibratable_params()
        for name, value in self.parameters.items():
            v = float(value)
            out.append(
                CalibratableParam(
                    process_name="",
                    field=f"parameters.{name}",
                    default=v,
                    clamp=default_clamp(v),
                    description=f"{self._param_label} {name!r} on {self._name}",
                )
            )
        return out
