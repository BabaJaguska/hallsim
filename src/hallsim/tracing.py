"""Whether a value is concrete, for code that needs a Python number from it.

``bool(x)``, ``float(x)`` and ``np.asarray(x)`` all raise when ``x`` is a JAX
tracer, and a function that does one of them works perfectly until the day
someone calls it inside ``jit`` / ``grad`` / ``vmap``. Anything reachable from
a traced run has to decide, explicitly, which it is:

- a **diagnostic** — a warning, a report, a screen — degrades: it has nothing
  concrete to inspect, so it stays quiet rather than raising inside someone's
  loss function.
- a **computation** whose result the caller depends on raises, naming what to
  resolve eagerly first. Silently returning something else is how a run
  produces a plausible wrong number.

Never let a diagnostic raise, and never let a computation guess.
"""

from __future__ import annotations

import jax


def is_traced(*values: object) -> bool:
    """True if any value is a JAX tracer, i.e. has no concrete value here."""
    return any(isinstance(v, jax.core.Tracer) for v in values)
