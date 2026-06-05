"""tellurium population timing — serial and multiprocess. JAX-free on purpose.

Run as a subprocess by demos/bench_population.py so its multiprocessing fork
never inherits the parent's CUDA context. Reads the same JSON config, prints
one JSON line to stdout: {"N": {"serial": s, "parallel": s, "nproc": k}, ...}.
"""
import json
import sys
import time

import numpy as np

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "configs/bench_population.json"
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
MODEL = cfg["model"]
T_END = float(cfg["t_end"])
N_SAVE = int(cfg["n_save"])
NS = list(cfg["n_sweep"])
SIGMA = float(cfg.get("perturb_sigma", 0.1))
SEED = int(cfg.get("seed", 0))
NPROC = int(cfg.get("nproc", 40))

# Worker state (one roadrunner per process), set by the Pool initializer.
_S = {}


def _init(model_path, t_end, n_save):
    import tellurium as te
    rr = te.loadSBMLModel(model_path)
    _S.update(rr=rr, fs=rr.model.getFloatingSpeciesIds(),
              t_end=t_end, n_save=n_save)


def _sim_chunk(factors_chunk):
    rr, fs, t_end, n = _S["rr"], _S["fs"], _S["t_end"], _S["n_save"]
    out = np.empty((len(factors_chunk), len(fs)))
    for i in range(len(factors_chunk)):
        rr.reset()
        for j, sp in enumerate(fs):
            k = f"[{sp}]"
            rr[k] = rr[k] * float(factors_chunk[i, j])
        res = rr.simulate(0, t_end, n)
        out[i] = res[-1, 1:1 + len(fs)]
    return out


def make_population(N, dim):
    return 1.0 + SIGMA * np.random.default_rng(SEED).standard_normal((N, dim))


def main():
    import multiprocessing as mp
    import tellurium as te

    rr0 = te.loadSBMLModel(MODEL)
    n_sp = len(rr0.model.getFloatingSpeciesIds())
    nproc = min(NPROC, mp.cpu_count())

    # One persistent fork pool, model loaded once per worker, reused across N.
    pool = mp.Pool(nproc, initializer=_init, initargs=(MODEL, T_END, N_SAVE))
    _init(MODEL, T_END, N_SAVE)  # also init this (serial) process

    results = {}
    for N in NS:
        fac = make_population(N, n_sp)
        t0 = time.time(); _sim_chunk(fac); serial = time.time() - t0
        k = min(nproc, N)
        chunks = [c for c in np.array_split(fac, k) if len(c)]
        pool.map(_sim_chunk, chunks)  # warm workers
        t0 = time.time(); pool.map(_sim_chunk, chunks); par = time.time() - t0
        results[str(N)] = {"serial": serial, "parallel": par, "nproc": k}
    pool.close(); pool.join()
    print(json.dumps(results))


if __name__ == "__main__":
    main()
