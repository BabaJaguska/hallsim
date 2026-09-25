"""The contrast interface calibration consumes, independent of modality.

A calibration problem never sees a measurement table. It sees group
contrasts: :meth:`MeasuredDataset.delta` for one comparison and
:meth:`MeasuredDataset.arm_deltas` for a time course whose reference
tracks each arm's. That arithmetic is the same whichever quantity was
measured, so a modality supplies only its log-scale values and, where
several measured features carry one quantity, how to collapse them.
"""

from __future__ import annotations

import pandas as pd


class MeasuredDataset:
    """Group contrasts over a ``quantity × sample`` table.

    A subclass provides :attr:`log_values` and ``sample_groups``. The index
    of :attr:`log_values` may repeat, for a modality that measures one
    quantity through several features; :meth:`_reduce` then says how the
    contrast collapses them, after the ratio rather than before it.
    """

    sample_groups: dict[str, list]

    @property
    def log_values(self) -> pd.DataFrame:
        """``quantity × sample`` on a log2 scale, so a fold change is a
        difference of group means."""
        raise NotImplementedError

    def _reduce(self, per_feature: pd.Series) -> pd.Series:
        return per_feature

    def _reduce_variance(self, per_feature: pd.Series) -> pd.Series:
        return per_feature

    def _columns(self, group: str) -> list:
        try:
            return self.sample_groups[group]
        except KeyError:
            raise KeyError(
                f"no sample group {group!r}; this dataset names "
                f"{sorted(self.sample_groups)}"
            ) from None

    def delta(self, condition: str, baseline: str) -> pd.Series:
        """Δ_data = log2 fold change between two named sample groups."""
        values = self.log_values
        cond = values[self._columns(condition)].mean(axis=1)
        base = values[self._columns(baseline)].mean(axis=1)
        return self._reduce(cond - base)

    def variance(self, condition: str, baseline: str) -> pd.Series:
        """Per-quantity sampling variance of the log2 fold change.

        ``Var(mean_cond − mean_base) = s²_cond/n_cond + s²_base/n_base``
        from replicate spread, counting only the samples that carry a
        value. Feed ``1/variance`` as ``weights`` to
        :class:`~hallsim.calibration.CalibrationProblem` to down-weight
        noisy quantities. With few replicates the estimate is itself noisy.
        """
        values = self.log_values
        cond = values[self._columns(condition)]
        base = values[self._columns(baseline)]
        return self._reduce_variance(
            cond.var(axis=1, ddof=1) / cond.count(axis=1)
            + base.var(axis=1, ddof=1) / base.count(axis=1)
        )

    def arm_deltas(
        self,
        samples: dict[str, dict[float, str]],
        arms: dict,
    ) -> dict[str, dict[float, pd.Series]]:
        """Log2 fold-change time courses whose reference matches each arm's.

        ``samples`` names the sample group for each arm at each time, e.g.
        ``{"DDIS_vs_ctrl": {0.0: "ETOPOSIDE_D00", 7.0: "ETOPOSIDE_D07"}}``;
        ``arms`` is the ``{name: Arm}`` the calibration problem takes, so
        the data contrast tracks the model's. An arm read against its own
        start is divided by its time-0 group, which then carries no data
        point; an arm read against another condition is divided by that
        condition's arm at the same time. An arm with no reference is not a
        fold change and is not built here.
        """
        arm_of_condition = {a.condition: name for name, a in arms.items()}
        out: dict[str, dict[float, pd.Series]] = {}
        for arm, by_time in samples.items():
            reference = arms[arm].reference
            if reference is None:
                raise ValueError(
                    f"arm {arm!r} has no reference, so its data are values, "
                    "not fold changes; read them from the sample groups "
                    "directly."
                )
            if reference == "t0":
                if 0.0 not in by_time:
                    raise ValueError(
                        f"arm {arm!r} reads against its own start but has "
                        "no time-0 group. Give it one, or reference another "
                        "condition."
                    )
                ref_for = {t: by_time[0.0] for t in by_time}
            else:
                base_arm = arm_of_condition.get(reference)
                if base_arm is None or base_arm not in samples:
                    raise ValueError(
                        f"arm {arm!r} references condition {reference!r}, "
                        "which no arm in `samples` supplies. A single-arm "
                        "dataset reads against its own start."
                    )
                ref_for = {
                    t: samples[base_arm][t]
                    for t in by_time
                    if t in samples[base_arm]
                }
            out[arm] = {
                t: self.delta(by_time[t], ref_for[t])
                for t in sorted(by_time)
                if t in ref_for and not (reference == "t0" and t == 0.0)
            }
        return out
