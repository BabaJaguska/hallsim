"""Post-fit reporting for a :class:`hallsim.calibration.CalibrationProblem`.

The output side of calibration — turns a fitted problem + its history into
the artifact bundle (topology graph, pre/post trajectory overlays,
trajectories + summary JSON). Kept separate from the optimizer loop so the
reporting concern doesn't sit next to the loss.
"""

from __future__ import annotations

import json
from pathlib import Path


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

    init = problem.initial_params()
    final = history.best_params or init

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
                "clamp": list(c.clamp) if c.clamp else None,
                "description": c.description,
            }
            for k, c in problem._coeffs.items()
        },
        "init_params": {k: float(v) for k, v in init.items()},
        "fitted_params": {k: float(v) for k, v in final.items()},
        "loss_history": [float(v) for v in history.losses],
        "val_loss_history": [float(v) for v in history.val_losses],
        "grad_norm_history": [float(v) for v in history.grad_norms],
        "lr_history": [float(v) for v in history.lrs],
        "lr_scale_history": [float(v) for v in history.lr_scales],
        "param_history": [
            {k: float(v) for k, v in dict(ph).items()}
            for ph in history.param_history
        ],
        "best_loss": float(history.best_loss),
        "stopped_epoch": history.stopped_epoch,
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


def rows_by_gene(result):
    """``{gene_symbol: row}`` for one concordance result."""
    return {r.reporter.gene_symbol: r for r in result.rows}


def format_table(pre, post, fit_arms=()) -> str:
    """Per-arm, per-timepoint out-of-the-box vs calibrated vs measured log2
    fold-changes, with the error, sign agreement and Spearman rho each arm
    moved by. Arms in ``fit_arms`` are labelled FIT, the rest HELD-OUT — pass
    ``problem.fit_arms``. Returns the table; the caller prints or writes it.
    """
    fit = set(fit_arms)
    out = [
        "=" * 74,
        "OUT-OF-THE-BOX vs CALIBRATED vs MEASURED  (log2 fold-change)",
        "=" * 74,
    ]
    for arm in pre:
        out.append(f"\n[{'FIT ' if arm in fit else 'HELD-OUT'}] {arm}")
        for t in sorted(pre[arm]):
            pre_r = rows_by_gene(pre[arm][t])
            post_r = rows_by_gene(post[arm][t])
            out.append(
                f"  day {t:g}   {'gene':<9}{'measured':>10}"
                f"{'model(oob)':>12}{'model(cal)':>12}   {'|err|oob→cal':>14}"
            )
            for g in pre_r:
                e0 = abs(pre_r[g].delta_sim - pre_r[g].delta_data)
                e1 = abs(post_r[g].delta_sim - post_r[g].delta_data)
                out.append(
                    f"  {'':<9}{g:<9}{pre_r[g].delta_data:>+10.3f}"
                    f"{pre_r[g].delta_sim:>+12.4f}"
                    f"{post_r[g].delta_sim:>+12.4f}   {e0:>6.3f}→{e1:<6.3f}"
                )
            out.append(
                f"  {'':<9}{'mean|err|':<9}{'':>10}{'':>12}{'':>12}   "
                f"{pre[arm][t].mean_abs_error:>6.3f}→"
                f"{post[arm][t].mean_abs_error:<6.3f}   "
                f"sign {pre[arm][t].sign_agreement * 100:.0f}→"
                f"{post[arm][t].sign_agreement * 100:.0f}%  "
                f"ρ {pre[arm][t].spearman_r:+.2f}→"
                f"{post[arm][t].spearman_r:+.2f}"
            )
    return "\n".join(out)


def plot_history(problem, history, path) -> None:
    """Loss curve, per-parameter trajectories, and gradient norm / effective
    LR over the fit. Parameters are drawn as their log-position within their
    clamp range, so knobs spanning orders of magnitude are comparable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(path)
    losses = np.asarray(history.losses)
    epochs = np.arange(1, len(losses) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(epochs, losses, color="#2a7")
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (log2FC MSE)")
    ax1.set_title("training loss")

    # Does a loss spike track a large gradient (→ clip) or an LR-schedule
    # event (→ plateau scheduler)?
    if history.grad_norms:
        gn = np.asarray(history.grad_norms)
        ax3.plot(epochs, gn, color="#c0392b", label="|grad| (global norm)")
        ax3.set_yscale("log")
        ax3.set_ylabel("|grad|", color="#c0392b")
        ax3.tick_params(axis="y", labelcolor="#c0392b")
        if history.lrs:
            axlr = ax3.twinx()
            axlr.plot(epochs, np.asarray(history.lrs), color="#2563eb")
            axlr.set_ylabel("effective LR", color="#2563eb")
            axlr.tick_params(axis="y", labelcolor="#2563eb")
        for e in epochs[np.r_[False, np.diff(losses) > 0.02]]:
            ax3.axvline(e, color="#999", lw=0.7, ls=":", zorder=0)
    ax3.set_xlabel("epoch")
    ax3.set_title("grad norm · LR scale  (dotted = loss spike)")

    # `clamp` is optional, so an unclamped parameter is normalized against
    # its own travelled range instead, dashed to keep the scales distinct.
    for name, ref in problem.param_refs.items():
        vals = np.asarray([float(ph[name]) for ph in history.param_history])
        clamped = ref.clamp is not None
        lo, hi = ref.clamp if clamped else (vals.min(), vals.max())
        if not (lo > 0 and hi > lo):
            ax2.plot(
                epochs,
                np.full_like(vals, 0.5),
                ls=":" if clamped else "--",
                label=f"{name} (flat)",
            )
            continue
        norm = (np.log(vals) - np.log(lo)) / (np.log(hi) - np.log(lo))
        ax2.plot(epochs, norm, ls="-" if clamped else "--", label=name)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("param (log-position in range)")
    ax2.set_title("parameter trajectories  (dashed = unclamped, own range)")
    ax2.legend(fontsize=7, loc="best")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


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
