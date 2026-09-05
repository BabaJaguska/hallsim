#!/usr/bin/env python
"""Cache one Jacobian per locally available SBML model.

    python scripts/build_jacobian_corpus.py [--only 157,318] [--list]

The stiffness verdict is a threshold on one number derived from a Jacobian, and
every claim about how to compute that number cheaply has so far been made on
synthetic matrices. This builds the real ones: import each model, wrap it with
:func:`hallsim.composite.single_process_composite`, and take
:func:`hallsim.bifurcation.jacobian` at the model's published initial condition
-- the same point :func:`hallsim.stiffness.analyze_groups` linearises at, so the
corpus matches shipped behaviour rather than inventing a new one.

Import dominates: sbmltoodejax codegen runs per call and is not cached, at
15-30 s a model, while the Jacobian itself is one ``jacfwd``. So the matrices go
to ``~/.cache/hallsim/jacobians/`` and every later analysis reads ``.npz``.

``rest_residual`` records how far the published IC is from a fixed point. A
large value means the Jacobian sits on a transient rather than an attractor,
which is worth knowing and is not worth "fixing" -- Newton on an autonomous
oscillator converges to the unstable point at the centre of its limit cycle.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

CACHE = Path.home() / ".cache" / "hallsim"
BIOMODELS = CACHE / "biomodels"
JACOBIANS = CACHE / "jacobians"
MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "jacobian_corpus"

#: Import errors and timeouts already measured; re-running them buys nothing.
SKIP = {56, 154, 155, 158, 446, 723, 1006, 252, 1044}

#: Vendored under demos/models/sbml/ and absent from the BioModels cache.
VENDORED = (
    "kowald2006/kowald2006_BIOMD0000000108.xml",
    "nazaret2009/nazaret2009_BIOMD0000000232.xml",
    "sivakumar2011/egf_394.xml",
    "sivakumar2011/shh_395.xml",
    "sivakumar2011/notch_396.xml",
    "sivakumar2011/wnt_397.xml",
    "sivakumar2011/crosstalk_398.xml",
)

TIMEOUT_S = 300

log = logging.getLogger("corpus")


def discover() -> list[tuple[str, str]]:
    """``(key, source)`` per model -- an accession or a path on disk."""
    out: list[tuple[str, str]] = []
    for f in sorted(BIOMODELS.glob("BIOMD*.xml")):
        acc = int(f.stem.replace("BIOMD", ""))
        if acc not in SKIP:
            # The cached path, not the accession: an accession re-enters the
            # BioModels fetch, and the corpus should not need the network.
            out.append((f"BIOMD{acc:010d}", str(f)))
    root = Path(__file__).resolve().parents[1] / "demos" / "models" / "sbml"
    for rel in VENDORED:
        p = root / rel
        if p.exists():
            out.append((p.stem, str(p)))
    return out


def build_one(key: str, source: str) -> dict:
    """Import, wrap, differentiate. Runs in its own process."""
    import numpy as np

    import hallsim  # noqa: F401  enables x64
    from hallsim.bifurcation import field_from_composite, jacobian
    from hallsim.composite import single_process_composite
    from hallsim.sbml_import import process_from_sbml

    t0 = time.time()
    proc = process_from_sbml(source, name="m")
    t_import = time.time() - t0

    comp = single_process_composite(proc, name="m")
    f, keys = field_from_composite(comp)
    y0 = np.asarray(comp.initial_state_vec(keys), dtype=float)

    t = time.time()
    J = np.asarray(jacobian(f, y0), dtype=float)
    t_jac = time.time() - t

    f0 = np.asarray(f(y0), dtype=float)
    ny = float(np.linalg.norm(y0))
    JACOBIANS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(JACOBIANS / f"{key}.npz", J=J, y0=y0,
                        keys=np.array(keys, dtype=object))
    return {
        "key": key,
        "source": source,
        "n_state": int(J.shape[0]),
        "native_time_seconds": float(getattr(proc, "native_time_seconds", 1.0)),
        "native_time_declared": bool(
            getattr(proc, "native_time_declared", False)),
        "rest_residual": float(np.linalg.norm(f0) / ny) if ny else float("nan"),
        "finite": bool(np.all(np.isfinite(J))),
        "t_import_s": t_import,
        "t_jacobian_s": t_jac,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated keys or accessions")
    ap.add_argument("--list", action="store_true", help="print models and exit")
    ap.add_argument("--worker", nargs=2, metavar=("KEY", "SOURCE"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s: %(message)s")

    if args.worker:  # one model, own process, so a crash cannot take the run
        print(json.dumps(build_one(*args.worker)))
        return

    models = discover()
    if args.only:
        want = {w.strip() for w in args.only.split(",")}
        models = [(k, s) for k, s in models
                  if k in want or s in want or k.lstrip("BIOMD0") in want]
    if args.list:
        for k, s in models:
            print(f"  {k:24s} {s}")
        print(f"  {len(models)} models")
        return

    MANIFEST.mkdir(parents=True, exist_ok=True)
    out = MANIFEST / "manifest.jsonl"
    done = set()
    if out.exists():  # resumable: a re-run skips what already landed
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    log.info("%d models, %d already cached", len(models), len(done))

    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "HALLSIM_COMPILATION_CACHE_DIR": "off"}
    with out.open("a") as fh:  # stream: a kill loses one model, not the run
        for i, (key, source) in enumerate(models, 1):
            if key in done:
                continue
            t = time.time()
            try:
                r = subprocess.run(
                    [sys.executable, __file__, "--worker", key, source],
                    capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
                if r.returncode != 0:
                    row = {"key": key, "source": source, "error":
                           (r.stderr.strip().splitlines() or ["?"])[-1][:300]}
                else:
                    row = json.loads(r.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                row = {"key": key, "source": source,
                       "error": f"timeout after {TIMEOUT_S}s"}
            except Exception as exc:  # a malformed worker line is data, not a stop
                row = {"key": key, "source": source,
                       "error": f"{type(exc).__name__}: {exc}"[:300]}
            row["t_total_s"] = time.time() - t
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            log.info("[%3d/%3d] %-24s %s", i, len(models), key,
                     f"n={row['n_state']}" if "n_state" in row
                     else row.get("error", "?")[:80])


if __name__ == "__main__":
    main()
