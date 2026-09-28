"""A PEtab problem's measurements as the contrast interface.

PEtab is the one source that deposits a model, the data it was fitted to and
the formula linking them, so its observables need no proxy: every other
modality reaches a model's states through a reporter or a regulon, and here
:attr:`PetabDataset.formulas` states each observable over the model's own
species.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hallsim.measurements import MeasuredDataset

log = logging.getLogger(__name__)

#: log2 per unit of each declared observable scale.
_TO_LOG2 = {"lin": None, "log": 1.0 / np.log(2.0), "log10": np.log2(10.0)}


def _group(condition: str, time: float) -> str:
    return f"{condition} @ {time:g}"


class PetabDataset(MeasuredDataset):
    """Group contrasts over a PEtab measurement table.

    A sample group is one simulation condition at one timepoint, named
    ``"<condition> @ <time>"``, so a course reads against its own start
    through :meth:`~hallsim.measurements.MeasuredDataset.arm_deltas` with a
    ``t0`` reference. Replicates of the same observable at the same condition
    and time become separate columns, which is what makes
    :meth:`~hallsim.measurements.MeasuredDataset.variance` meaningful.
    """

    def __init__(
        self,
        log_values: pd.DataFrame,
        sample_groups: dict[str, list],
        *,
        formulas: dict[str, str],
        design=None,
        conditions: dict[str, str] | None = None,
        name: str = "",
    ):
        self._log = log_values
        self.sample_groups = sample_groups
        self.formulas = formulas
        self.design = design
        #: ``{conditionId: arm name}``. A group is keyed by the name, so a
        #: caller holding the id the measurement table uses is resolved here
        #: rather than left to guess which of the two names a group.
        self.conditions = conditions or {}
        self.name = name

    def arm(self, condition: str) -> str:
        """The arm a condition id or name refers to."""
        return self.conditions.get(condition, condition)

    @property
    def log_values(self) -> pd.DataFrame:
        return self._log

    @classmethod
    def from_problem(cls, problem, name: str = "") -> "PetabDataset":
        """Build from a ``petab.v1.Problem``."""
        from petab.v1.C import (
            LIN,
            MEASUREMENT,
            OBSERVABLE_FORMULA,
            OBSERVABLE_ID,
            OBSERVABLE_TRANSFORMATION,
            SIMULATION_CONDITION_ID,
            TIME,
        )

        from hallsim.search.attached import condition_names, petab_design

        m = problem.measurement_df
        if m is None or m.empty:
            raise ValueError("the problem carries no measurements")
        obs = problem.observable_df
        conditions = problem.condition_df

        scale = {}
        formulas = {}
        if obs is not None:
            if OBSERVABLE_TRANSFORMATION in obs:
                scale = dict(obs[OBSERVABLE_TRANSFORMATION].fillna(LIN))
            if OBSERVABLE_FORMULA in obs:
                formulas = {
                    str(k): str(v) for k, v in obs[OBSERVABLE_FORMULA].items()
                }

        names = condition_names(
            conditions.reset_index() if conditions is not None else None
        )

        rows = m.copy()
        rows[TIME] = pd.to_numeric(rows[TIME], errors="coerce")
        if SIMULATION_CONDITION_ID in rows:
            arms = [
                str(names.get(c) or c or "")
                for c in rows[SIMULATION_CONDITION_ID]
            ]
        else:
            arms = [""] * len(rows)
        rows["_arm"] = arms
        rows["_group"] = [_group(a, t) for a, t in zip(arms, rows[TIME])]
        # A replicate is a second row for the same observable in the same
        # group; ranking within the group gives each its own column, so the
        # spread survives into `variance`.
        rows["_rep"] = rows.groupby([OBSERVABLE_ID, "_group"]).cumcount()
        rows["_col"] = rows["_group"] + "#" + rows["_rep"].astype(str)

        values = rows.pivot_table(
            index=OBSERVABLE_ID,
            columns="_col",
            values=MEASUREMENT,
            aggfunc="first",
        )

        factor = pd.Series(
            {
                o: _TO_LOG2.get(str(scale.get(o, LIN)), None)
                for o in values.index
            }
        )
        linear = values[factor.isna()]
        logged = values[factor.notna()]
        parts = []
        if not linear.empty:
            with np.errstate(divide="ignore", invalid="ignore"):
                parts.append(np.log2(linear.where(linear > 0)))
        if not logged.empty:
            parts.append(logged.mul(factor[logged.index], axis=0))
        log_values = pd.concat(parts).reindex(values.index)

        groups: dict[str, list] = {}
        for group, column in zip(rows["_group"], rows["_col"]):
            if column not in groups.setdefault(group, []):
                groups[group].append(column)
        return cls(
            log_values,
            groups,
            formulas=formulas,
            design=petab_design(
                m,
                conditions.reset_index() if conditions is not None else None,
            ),
            conditions={str(k): str(v) for k, v in names.items()},
            name=name,
        )

    @classmethod
    def from_yaml(cls, url: str, name: str = "") -> "PetabDataset":
        """Build from a PEtab YAML, local or remote."""
        from hallsim.search.attached import load_problem

        return cls.from_problem(load_problem(url), name=name or str(url))

    @classmethod
    def from_benchmark(cls, problem: str) -> "PetabDataset":
        """Build from a problem in the PEtab benchmark collection by name."""
        from hallsim.search.attached import PETAB_RAW

        base = f"{PETAB_RAW}/Benchmark-Models/{problem}"
        return cls.from_yaml(f"{base}/{problem}.yaml", name=problem)

    def times(self, condition: str | None = None) -> list[float]:
        """The timepoints a condition carries, in order. Takes its id or its
        name."""
        want = self.arm(condition) if condition is not None else None
        out = set()
        for group in self.sample_groups:
            arm, _, time = group.rpartition(" @ ")
            if want is None or arm == want:
                out.add(float(time))
        return sorted(out)

    def course(self, condition: str) -> dict[float, str]:
        """``{time: group}`` for one condition, ready for ``arm_deltas``."""
        arm = self.arm(condition)
        times = self.times(arm)
        if not times:
            raise KeyError(
                f"no condition {condition!r}; this problem names "
                f"{sorted({g.rpartition(' @ ')[0] for g in self.sample_groups})}"
            )
        return {t: _group(arm, t) for t in times}
