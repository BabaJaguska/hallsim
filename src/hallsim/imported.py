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

from hallsim.process import Process


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

    # Label for this format's parameters in calibration descriptions,
    # e.g. "SBML constant" / "XPP parameter". Class attribute, not a field.
    _param_label = "parameter"

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
