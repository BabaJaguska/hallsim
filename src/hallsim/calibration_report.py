"""Post-fit reporting for a :class:`hallsim.calibration.CalibrationProblem`.

The output side of calibration — turns a fitted problem + its history into
the artifact bundle (topology graph, pre/post trajectory overlays,
trajectories + summary JSON). Kept separate from the optimizer loop so the
reporting concern doesn't sit next to the loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp


def save_outputs(
    problem,
    out_dir: str,
    history,
    *,
    n_save_plot: int = 500,
) -> dict:
    """Produce the post-fit artifact bundle for ``problem`` in ``out_dir``.

    Generates:

    - ``graph.png`` — composite topology rendered via networkx.
    - ``trajectories_<cond>_pre_vs_post.png`` — one figure per condition
      overlaying pre-fit and post-fit reporter trajectories.
    - ``trajectories_post_all_arms.png`` — all conditions at post-fit.
    - ``trajectories.json`` — per-condition reporter-path trajectories at
      post-fit (densely sampled, ``n_save_plot`` points).
    - ``summary.json`` — fitted params, init params, loss history, per-arm
      concordance (pre and post), conditions, params.

    Re-samples each condition at ``n_save_plot`` points so the trajectory
    plots are smooth (the loss path uses ``problem.n_save``, kept low for
    jvp tractability). Returns a dict describing the written artifacts.
    """
    from hallsim.plotting import (
        draw_composite_graph,
        plot_runs_comparison,
        save_run_results,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    init = {k: jnp.asarray(p.init) for k, p in problem._all_refs.items()}
    final = history.final_params or init

    # Densely-sampled trajectories at both ends of the fit.
    pre_runs = problem.simulate_all_conditions(init, n_save=n_save_plot)
    post_runs = problem.simulate_all_conditions(final, n_save=n_save_plot)

    # Concordance — uses the standard n_save path (matches the numbers the
    # demo prints) so the JSON tallies with stdout.
    results_pre = problem.evaluate(init)
    results_post = problem.evaluate(final)

    reporter_paths = [r.observable for r in problem.reporters]

    # 1. Topology
    draw_composite_graph(
        problem.composite,
        save=str(out / "graph.png"),
        title="composite topology",
    )

    # 2. Per-condition pre-vs-post trajectory overlays
    for cond_name in problem.conditions:
        plot_runs_comparison(
            {
                "pre-fit": pre_runs[cond_name],
                "post-fit": post_runs[cond_name],
            },
            paths=reporter_paths,
            title=f"{cond_name}: pre vs post",
            save=str(out / f"trajectories_{cond_name}_pre_vs_post.png"),
        )

    # 3. All conditions at post-fit
    plot_runs_comparison(
        post_runs,
        paths=reporter_paths,
        title="all conditions at post-fit params",
        save=str(out / "trajectories_post_all_arms.png"),
    )

    # 4. Trajectories JSON (post-fit only — pre-fit is in the plots)
    save_run_results(
        post_runs,
        str(out / "trajectories.json"),
        paths=reporter_paths,
        metadata={
            "fitted_params": {k: float(v) for k, v in final.items()},
            "n_save_plot": n_save_plot,
            "t_end": problem.t_end,
            "macro_dt": problem.macro_dt,
        },
    )

    # 5. Summary JSON
    summary = {
        "params": {
            k: {
                "process_name": p.process_name,
                "field": p.field,
                "init": p.init,
                "clamp": list(p.clamp) if p.clamp else None,
                "description": p.description,
            }
            for k, p in problem.params.items()
        },
        "hallmark_coeffs": {
            k: {
                "hallmark": c.hallmark,
                "param_name": c.param_name,
                "coeff": c.coeff,
                "init": c.init,
                "clamp": list(c.clamp) if c.clamp else None,
                "description": c.description,
            }
            for k, c in problem._coeffs.items()
        },
        "init_params": {k: float(v) for k, v in init.items()},
        "fitted_params": {k: float(v) for k, v in final.items()},
        "loss_history": [float(v) for v in history.losses],
        "wall_time_s": float(history.wall_time_s),
        "conditions": {
            name: {
                "hallmarks": dict(c.hallmarks),
                "description": c.description,
            }
            for name, c in problem.conditions.items()
        },
        "arm_pairs": dict(problem.arm_pairs),
        "fit_arms": list(problem.fit_arms),
        "held_out_arms": list(problem.held_out_arms),
        "t_end": problem.t_end,
        "macro_dt": problem.macro_dt,
        "concordance_pre": _conc_to_dict(results_pre),
        "concordance_post": _conc_to_dict(results_post),
        "reporters": [
            {
                "gene_symbol": r.gene_symbol,
                "observable": r.observable,
                "sign": r.sign,
                "summary": (
                    r.summary.__name__
                    if hasattr(r.summary, "__name__")
                    else type(r.summary).__name__
                ),
            }
            for r in problem.reporters
        ],
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "out_dir": str(out),
        "files": sorted(p.name for p in out.iterdir()),
    }


def _conc_to_dict(results_dict: dict) -> dict:
    out: dict = {}
    for arm, per_t in results_dict.items():
        out[arm] = {}
        for t, r in per_t.items():
            out[arm][f"{t:g}"] = {
                "timepoint": float(t),
                "sign_agreement": float(r.sign_agreement),
                "spearman_r": float(r.spearman_r),
                "n_compared": r.n_compared,
                "rows": [
                    {
                        "gene": row.reporter.gene_symbol,
                        "observable": row.reporter.observable,
                        "delta_sim_signed": float(row.delta_sim),
                        "delta_data": float(row.delta_data),
                        "sign_match": bool(row.sign_match),
                    }
                    for row in r.rows
                ],
            }
    return out
