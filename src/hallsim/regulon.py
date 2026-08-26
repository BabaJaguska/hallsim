"""Regulon readout head — modelled TF activity to transcriptome-wide log2FC.

:mod:`hallsim.gene_reporters` maps one mechanistic observable to one textbook
transcript by hand, which caps a composite's transcriptomic readout at the
handful of genes someone curated. This is the scaled readout: a signed
TF→target prior expands the regulator activities a composite already carries
across every gene those regulators are known to touch.

    Δlog2 x̂ = (S ⊙ W) · Δa

``S`` is the {-1, 0, +1} CollecTRI prior over (gene × TF), ``Δa`` the modelled
activity deltas, ``W`` the fitted gains. Gains are parameterized as ``exp`` so
they stay positive: the *sign* of every prediction is then fixed by the prior
alone, making sign concordance a zero-parameter prediction and leaving only
magnitude to fit.

Unlike the canonical reporters this layer *is* fitted, so it is a model rather
than a validation instrument — score it on held-out perturbations and report it
separately from reporter concordance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

log = logging.getLogger(__name__)


def _reference_dir() -> Path:
    """Vendored reference-table root, or ``HALLSIM_DATA_DIR`` when set."""
    env = os.environ.get("HALLSIM_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parent / "reference"


def load_collectri(path: Path | None = None) -> pd.DataFrame:
    """The vendored CollecTRI human prior as ``source/target/weight``.

    Müller-Dott et al., NAR 2023, retrieved through OmniPath. ``weight`` is
    +1 for activation and -1 for repression.
    """
    if path is None:
        path = _reference_dir() / "collectri" / "collectri_human.tsv"
    return pd.read_csv(path, sep="\t")


@dataclass(frozen=True)
class Regulon:
    """A signed TF→target prior restricted to the TFs a composite models.

    ``signs`` is ``(n_genes, n_tfs)`` in {-1, 0, +1}, aligned to ``genes`` and
    ``tfs``. Build it with :meth:`from_collectri` and hand it to
    :class:`RegulonHead`.
    """

    tfs: tuple[str, ...]
    genes: tuple[str, ...]
    signs: np.ndarray

    @classmethod
    def from_collectri(
        cls,
        tfs: list[str],
        prior: pd.DataFrame | None = None,
        restrict_to: list[str] | None = None,
    ) -> "Regulon":
        """Prior restricted to ``tfs``, over the union of their targets.

        ``restrict_to`` further intersects the gene axis with a measured gene
        list, so the head only predicts genes the data can score.
        """
        if prior is None:
            prior = load_collectri()
        sub = prior[prior["source"].isin(tfs)]
        missing = set(tfs) - set(sub["source"])
        if missing:
            log.warning("TFs absent from the prior: %s", sorted(missing))

        genes = sorted(set(sub["target"]))
        if restrict_to is not None:
            genes = sorted(set(genes) & set(restrict_to))
        if not genes:
            raise ValueError("regulon is empty after restriction")

        gene_ix = {g: i for i, g in enumerate(genes)}
        tf_ix = {t: j for j, t in enumerate(tfs)}
        signs = np.zeros((len(genes), len(tfs)))
        for tf, gene, w in zip(sub["source"], sub["target"], sub["weight"]):
            i = gene_ix.get(gene)
            if i is not None:
                signs[i, tf_ix[tf]] = np.sign(w)
        return cls(tuple(tfs), tuple(genes), signs)

    @property
    def coverage(self) -> dict[str, int]:
        """Targets carried per TF, plus the union size."""
        per_tf = {
            tf: int((self.signs[:, j] != 0).sum())
            for j, tf in enumerate(self.tfs)
        }
        per_tf["__union__"] = len(self.genes)
        return per_tf


class RegulonHead(eqx.Module):
    """Δ activity → Δlog2 expression, one fitted gain per TF.

    ``signs`` is the prior, held as a constant array; ``log_gain`` is the only
    trainable state, so the head adds ``n_tfs`` parameters regardless of how
    many genes it predicts. Call it on ``(..., n_tfs)`` deltas to get
    ``(..., n_genes)`` predictions — batched over conditions, no Python loop.
    """

    signs: jnp.ndarray
    log_gain: jnp.ndarray

    def __init__(self, regulon: Regulon, init_gain: float = 1.0):
        self.signs = jnp.asarray(regulon.signs)
        self.log_gain = jnp.full(len(regulon.tfs), jnp.log(init_gain))

    def __call__(self, delta_activity: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.asarray(delta_activity)
            @ (self.signs * jnp.exp(self.log_gain)).T
        )

    def predict_signs(self, delta_activity: jnp.ndarray) -> jnp.ndarray:
        """Predicted direction only — independent of ``log_gain`` because
        gains are positive, so this is a zero-parameter prediction."""
        return jnp.sign(self(delta_activity))


def fit_gains(
    head: RegulonHead,
    delta_activity: jnp.ndarray,
    observed: jnp.ndarray,
    steps: int = 500,
    lr: float = 0.05,
) -> tuple[RegulonHead, float]:
    """Fit the per-TF gains by gradient descent on mean squared log2FC error.

    ``delta_activity`` is ``(n_cond, n_tfs)`` and ``observed`` ``(n_cond,
    n_genes)`` on the head's gene axis. Returns the fitted head and its final
    loss. Only ``log_gain`` moves — the prior is fixed.
    """
    da = jnp.asarray(delta_activity)
    obs = jnp.asarray(observed)
    signs = head.signs

    def loss(log_gain):
        pred = da @ (signs * jnp.exp(log_gain)).T
        return jnp.mean((pred - obs) ** 2)

    opt = optax.adam(lr)
    params = head.log_gain
    state = opt.init(params)
    grad_fn = jax.jit(jax.value_and_grad(loss))

    value = jnp.inf
    for _ in range(steps):
        value, grads = grad_fn(params)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)

    return eqx.tree_at(lambda h: h.log_gain, head, params), float(value)


@dataclass(frozen=True)
class RegulonScore:
    """Genome-wide agreement between a predicted and a measured log2FC."""

    n_genes: int
    sign_agreement: float
    spearman: float
    pearson: float

    def __str__(self) -> str:
        return (
            f"n={self.n_genes}  sign={self.sign_agreement:.3f}  "
            f"rho={self.spearman:.3f}  r={self.pearson:.3f}"
        )


def score_predictions(
    predicted: np.ndarray,
    observed: np.ndarray,
    min_abs_observed: float = 0.0,
) -> RegulonScore:
    """Sign agreement and rank/linear correlation over the flattened pair.

    ``min_abs_observed`` drops genes whose measured change is below a
    magnitude, so sign agreement is not dominated by genes that did not move.
    """
    from scipy.stats import pearsonr, spearmanr

    pred = np.asarray(predicted).ravel()
    obs = np.asarray(observed).ravel()
    keep = (np.abs(obs) >= min_abs_observed) & (pred != 0)
    pred, obs = pred[keep], obs[keep]
    if pred.size < 3:
        raise ValueError("too few genes survive the magnitude filter to score")

    return RegulonScore(
        n_genes=int(pred.size),
        sign_agreement=float(np.mean(np.sign(pred) == np.sign(obs))),
        spearman=float(spearmanr(pred, obs).statistic),
        pearson=float(pearsonr(pred, obs).statistic),
    )
