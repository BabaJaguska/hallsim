"""One-way hybrid multi-hallmark run with Proctor 2007 simulated by SSA.

DP14 and GZ06 are integrated deterministically through ``Scheduler``. Their
trajectory drives Proctor's ROS input and synthesis-rate input while Proctor's
reaction network is advanced with direct Gillespie SSA. This is an explicit
one-way hybrid, not a claim that the full composite has a single global SSA
semantics.
"""

from __future__ import annotations

import numpy as np

from demos.models.multi_hallmark import (
    DP14_MTORC1_ACTIVE_NAME,
    DP14_ROS_NAME,
    PROCTOR07_ROS_NAME,
    _add_proteostasis,
    build_multi_hallmark_composite,
)
from hallsim.scheduler import Scheduler
from hallsim.stochastic import SSAResult, simulate_ssa


def run(
    *,
    t_end: float = 14.0,
    save_dt: float = 0.1,
    seed: int = 0,
    max_events: int = 10_000_000,
) -> tuple[SSAResult, object]:
    """Run deterministic DP14/GZ06 and stochastic Proctor on one-way inputs."""
    deterministic = build_multi_hallmark_composite(validate=False)
    keys = deterministic.store_keys()
    trajectory = Scheduler().run(
        deterministic,
        t_span=(0.0, t_end),
        macro_dt=1.0,
        save_dt=save_dt,
        y0=deterministic.initial_state_vec(keys),
    )
    key_index = {key: i for i, key in enumerate(keys)}

    processes = {"dp14": deterministic.processes["dp14"]}
    topology = {}
    _add_proteostasis(processes, topology, processes["dp14"])
    p07 = processes["p07"].as_stochastic()
    processes["p07"] = p07
    ros_edge = processes["ros_identity"]
    mtor_edge = processes["mtor_synthesis"]

    def input_provider(t, state):
        ros = np.interp(
            t,
            np.asarray(trajectory.ts),
            np.asarray(trajectory.ys[..., key_index[f"dp14/{DP14_ROS_NAME}"]]),
        )
        mtor = np.interp(
            t,
            np.asarray(trajectory.ts),
            np.asarray(
                trajectory.ys[
                    ..., key_index[f"dp14/{DP14_MTORC1_ACTIVE_NAME}"]
                ]
            ),
        )
        return {
            PROCTOR07_ROS_NAME: float(ros_edge.gain) * float(ros),
            "k1_in": float(mtor_edge.offset)
            + float(mtor_edge.gain) * float(mtor),
        }

    result = simulate_ssa(
        p07,
        t_span=(0.0, t_end),
        save_dt=save_dt,
        seed=seed,
        max_events=max_events,
        input_provider=input_provider,
    )
    return result, trajectory


if __name__ == "__main__":
    result, _ = run()
    print(
        f"SSA events={result.reaction_indices.size} "
        f"samples={len(result.times)}"
    )
