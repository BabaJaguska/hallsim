# Formalism coverage

Computational biology slides love to show ~13 modeling formalisms
(Differential Equations, Gillespie, FBA, Boolean networks, Molecular
Dynamics, Agent-based modeling, Brownian/Stokesian, …) all converging
on an "Integrated Multiscale Model." HallSim does **not** cover all of
them. This doc is a frank accounting of what's in scope, what's
roadmap, and what's a different kind of tool.

## The deliberate trade

HallSim's composition contract — `Process.derivative` /
`Process.update` / `Process.condition + handler`, all consuming and
producing JAX arrays — is **narrower than Vivarium's** by design.
Vivarium's Engine accepts any Python callable behind an `update(t, dt)`
interface, so it can compose FBA + ODE + agent-based via wrappers like
COBRApy + Tellurium + MoBio. That breadth comes with a cost: the
composite is opaque to JAX, so you cannot:

- JIT-compile the joined system
- `jax.grad` across formalism boundaries
- batch populations on a GPU without per-element Python overhead

HallSim restricts the contract to JAX-expressible computation. In
exchange, the **whole composite is differentiable, JIT-compilable,
GPU-runnable, and natively batched**. For aging biology — where most
of the action is in signaling networks, metabolic ODEs, oscillators
(p53–Mdm2), and stem-cell niches — the trade is overwhelmingly
favorable. For genome-scale FBA or molecular dynamics it is not.

### Bigraph framing — kept the link graph, dropped the place graph

Vivarium-collective papers cite Milner's bigraphs (place graph for
nesting + link graph for connections) as the formal foundation of their
topology language. HallSim **adopts the link-graph half and deliberately
drops the place graph** in exchange for a single-tensor JIT-friendly
state. Specifically:

- **Link graph (kept).** Our `topology = {process: {port: store_path}}`
  is the link graph: which processes connect to which store paths. A
  process can write to multiple stores (multiple ports), and a store
  can be read/written by multiple processes (additive EVOLVED
  semantics). Same picture Vivarium implements.
- **Place graph (dropped).** Vivarium nests stores hierarchically as
  Python dicts with `..`-style path traversal. We use a *flat*
  `dict[str, jnp.ndarray]` state with `/`-separated string keys
  (`"cytoplasm/ROS"`) — convention, not nesting.

**Why this is not a loss for multi-cell modelling.** A flat keyspace
plus per-cell *batching* on a trailing axis (already supported by
`Scheduler.run` via batched y0) covers the cases where Vivarium's
hierarchy buys real expressivity:

- *Homogeneous N-cell populations*: pass a `(batch, n_vars)` y0; one
  JIT-compiled run produces N independent cell trajectories.
- *Inter-cell paracrine coupling*: a `PopulationAggregate` Process
  reads along the batch axis (e.g. `mean(IL6) across cells`) and writes
  a shared `env/IL6` store path that every cell reads back as INPUT.
  Same machinery as Vivarium's place-graph cross-level wiring; no
  hierarchy required.
- *Re-using the same composite under two prefixes* (e.g. ERiQ in
  cytoplasm and a hypothetical nuclear analog): handled by a small
  `re_namespace(composite, prefix)` helper plus a `union(comps)`
  operator. ~20 lines total. Not implemented yet because no model
  needs it; trivially added when one does.

Net: Vivarium's place-graph + link-graph framing is a useful conceptual
scaffold for hierarchical composition, but a flat `jnp.ndarray` state
with topology-only wiring is computationally superior — one JIT-compiled
tensor, native batching along the trailing axis, no path-traversal
overhead. The convenience of one-liner sub-composite embedding is
recoverable via `re_namespace` without reintroducing nesting.

## Coverage table

| Formalism | Status | Notes |
|---|---|---|
| Differential Equations | ✅ | Native CONTINUOUS Process via Diffrax |
| Michaelis-Menten kinetics | ✅ | One lens on enzyme kinetics; ODE form, exactly the ERiQ revisions |
| Monod-Wyman-Changeux | ✅ | Algebraic + DE; allosteric receptors fall out of the same machinery |
| Poisson Process (degradation) | ✅ | Linear DE limit; CONTINUOUS Process |
| Neural Network | ✅ | `NeuralODE` Process with training infrastructure (`hallsim.models.neuralode`) |
| SBML — deterministic ODE | 🟡 | `process_from_sbml` via `sbmltoodejax`; pre-imported models in `models/sivakumar2011/`, `models/zatorsky2006/` |
| SBML — events | 🟡 | `sbmltoodejax` skips them; HallSim has `ProcessKind.EVENT` and Diffrax 0.5+ supports events natively. Missing piece is the SBML event-MathML translator (roadmap) |
| Boolean Network | 🟡 | Trivial as DISCRETE (`jnp.where` / logical ops); abstraction supports it; no model written |
| Gillespie / Stochastic | 🟡 | Would be stochastic DISCRETE with `jax.random` keys threaded through Scheduler. Diffrax also supports SDEs natively. No model written — see [Roadmap](../README.md#stochastic-discrete--gillespie-support) |
| Rule-based (BNGL / Kappa) | 🟡 | Could be a CONTINUOUS Process emitting an auto-expanded ODE system; not built |
| Agent-based / multi-cellularity | 🟡 | Batched y0 gives N independent cells; inter-cell communication (paracrine, diffusion) needs a `PopulationAggregate` abstraction — see [Roadmap](../README.md#multi-cell--inter-cell-communication) |
| Constraint-based optimization (FBA / BiGG) | ❌ | Real gap. Genome-scale metabolic flux balance needs an LP solver. Doable via `jaxopt` (which has implicit-differentiation LP/QP solvers) but not wired in. Coupling ERiQ signaling to genome-scale flux is a future paper, not a preprint feature |
| Molecular Dynamics | ❌ | Wrong scale. Use OpenMM / GROMACS |
| Brownian / Stokesian dynamics | ❌ | Diffrax supports SDEs, but you'd need spatial state representation that HallSim doesn't have |
| Physics Engine | ❌ | Out of scope |
| Graphical Model / Bayesian network | ❌ | Different paradigm. Could be wrapped behind a Process but not part of the dynamic-systems core |

## How to read this table

✅ — the formalism is either native or reduces to something native;
working models exist.

🟡 — the framework's abstractions support it, but no canonical example
exists yet. Adding one is mostly mechanical Process authoring, not
architectural work.

❌ — outside HallSim's design envelope. For these you compose
externally (run FBA in COBRApy, MD in OpenMM, etc.) and feed boundary
fluxes/state into a HallSim Process as INPUT ports. Vivarium-style
"everything in one engine" composition is not the goal here;
"everything that's JAX-expressible, composed under one differentiable
runner" is.

## On the "am I making a mutilated Vivarium?" question

No. Vivarium's universal-integrator framing is real, but it ignores
that **none of those formalisms compose under autodiff**. HallSim's
contribution is not breadth of formalism support; it's
end-to-end-differentiable composition within a deliberately chosen
subset. That subset happens to cover the formalisms most relevant to
network-scale aging biology.

If your digital-twin ambition outgrows the subset (you genuinely need
genome-scale FBA coupled to spatial agent-based tissue), the answer is
not "HallSim should add FBA"; it's "HallSim composes with COBRApy at
its boundary, and the genome-scale flux state enters as an INPUT
port." Same pattern as SBML import: don't reimplement, interoperate at
the well-defined interface.
