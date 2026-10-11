"""The contrast interface calibration consumes, independent of modality.

A calibration problem never sees a measurement table. It sees contrasts:
:meth:`MeasuredDataset.delta` between two sample groups,
:meth:`MeasuredDataset.member_delta` per subject where the same subjects
were sampled twice, and :meth:`MeasuredDataset.arm_deltas` for a time
course whose reference tracks each arm's, in either form. That arithmetic
is the same whichever quantity was measured, so a modality supplies only
its log-scale values and, where several measured features carry one
quantity, how to collapse them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pandas as pd


class MeasuredDataset:
    """Group contrasts over a ``quantity × sample`` table.

    A subclass provides :attr:`log_values` and ``sample_groups``. The index
    of :attr:`log_values` may repeat, for a modality that measures one
    quantity through several features; :meth:`_reduce` then says how the
    contrast collapses them, after the ratio rather than before it.
    """

    sample_groups: dict[str, list]

    #: ``{sample: subject}`` — which person, animal or culture each sample
    #: was drawn from, where the deposit says. Empty when it does not, and
    #: then only group contrasts exist.
    sample_members: Mapping[str, str] = MappingProxyType({})

    #: Namespace :attr:`log_values` is indexed in, so a species carrying the
    #: same identifier joins to it. Empty when the index is not identifiers.
    measures: str = ""

    @property
    def measured(self) -> pd.Index:
        """The distinct quantities measured, keyed as :attr:`measures`."""
        return self.log_values.index.unique()

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
        """Δ_data = log2 fold change between two named sample groups.

        A quantity the contrast cannot determine — measured in neither
        group, or below detection throughout the baseline — is absent from
        the result rather than present as a missing value, because a
        calibration target that is not a number is not a target.
        """
        values = self.log_values
        cond = values[self._columns(condition)].mean(axis=1)
        base = values[self._columns(baseline)].mean(axis=1)
        return self._reduce(cond - base).dropna()

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

    def _by_member(self, group: str) -> pd.DataFrame:
        """``quantity × subject`` log values for one group, technical
        replicates of a subject averaged."""
        columns = self._columns(group)
        missing = [s for s in columns if s not in self.sample_members]
        if missing:
            raise ValueError(
                f"{len(missing)} samples in group {group!r} name no subject "
                f"(e.g. {missing[0]!r}); a per-subject contrast needs "
                "`sample_members` for every sample"
            )
        values = self.log_values[columns]
        subjects = [self.sample_members[s] for s in columns]
        return values.T.groupby(subjects).mean().T

    def member_delta(
        self,
        condition: str,
        baseline: str,
        members: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Δ_data per subject: ``subject × quantity`` log2 fold change of
        each subject's ``condition`` sample against its own ``baseline``.

        This is the paired contrast, which keeps between-subject variation
        that :meth:`delta` averages away. ``members`` fixes the rows and
        their order — a calibration batch is positional — and a subject
        lacking either sample is a row of missing values, which the loss
        masks. Without ``members`` the rows are the subjects present in
        both groups, sorted.
        """
        if not self.sample_members:
            raise ValueError(
                "this dataset does not say which subject each sample came "
                "from, so only group contrasts exist; build it with "
                "`sample_members`"
            )
        cond = self._by_member(condition)
        base = self._by_member(baseline)
        if members is None:
            members = sorted(set(cond.columns) & set(base.columns))
            if not members:
                raise ValueError(
                    f"no subject was sampled in both {condition!r} and "
                    f"{baseline!r}; the groups are unpaired, so use `delta`"
                )
        members = list(members)
        diff = cond.reindex(columns=members) - base.reindex(columns=members)
        reduced = diff.apply(self._reduce)
        return reduced.T.loc[:, reduced.notna().any(axis=1).values]

    def arm_deltas(
        self,
        samples: dict[str, dict[float, str]],
        arms: dict,
        members: Sequence[str] | bool | None = None,
    ) -> dict[str, dict[float, pd.Series | pd.DataFrame]]:
        """Log2 fold-change time courses whose reference matches each arm's.

        ``samples`` names the sample group for each arm at each time, e.g.
        ``{"DDIS_vs_ctrl": {0.0: "ETOPOSIDE_D00", 7.0: "ETOPOSIDE_D07"}}``;
        ``arms`` is the ``{name: Arm}`` the calibration problem takes, so
        the data contrast tracks the model's. An arm read against its own
        start is divided by its time-0 group, which then carries no data
        point; an arm read against another condition is divided by that
        condition's arm at the same time. An arm with no reference is not a
        fold change and is not built here.

        ``members`` asks for the paired form: each timepoint is a
        ``subject × quantity`` frame from :meth:`member_delta`, the shape a
        calibration condition with a batched start takes, one row per
        member. Pass the subjects in the order of that batch, or ``True``
        for every subject the arm's reference group holds, sorted. The rows
        are the same at every timepoint, so a subject missing one sample is
        a row of missing values there rather than a shifted batch.
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
            times = [
                t
                for t in sorted(by_time)
                if t in ref_for and not (reference == "t0" and t == 0.0)
            ]
            if members is None or members is False:
                out[arm] = {
                    t: self.delta(by_time[t], ref_for[t]) for t in times
                }
                continue
            rows = (
                sorted(
                    {
                        self.sample_members[s]
                        for g in set(ref_for.values())
                        for s in self._columns(g)
                        if s in self.sample_members
                    }
                )
                if members is True
                else list(members)
            )
            out[arm] = {
                t: self.member_delta(by_time[t], ref_for[t], rows)
                for t in times
            }
        return out
