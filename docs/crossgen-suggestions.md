# Cross-Domain Suggestions for HallSim Multi-Scale Coupling

Actionable ideas extracted from three CrossGen sessions (April 2025).

HallSim's Scheduler splitting schemes (Strang splitting, interpolated
dense-output coupling, adaptive macro_dt) and multi-scale roadmap were
informed by these sessions, generated with
[CrossGen](https://github.com/mohamedAtoui/CrossGen) — a structure-mapping
tool that surfaces solutions from unrelated fields. Several turned out to be
established methods in those communities (gyrokinetic plasma simulation,
partitioned fluid-structure interaction, Mori-Zwanzig projection, waveform
relaxation, federated Kalman filtering) for the same coupling problems.

**Session 1** (biology-default lenses): immunological germinal centers
and mycorrhizal fungal networks. Mostly standard numerical methods in
biological costume, but the mycorrhizal interpolant idea was real.

**Session 2** (`--prefer-stem --skip-biologize`): federated Kalman
filtering, polyphase filterbanks, waveform relaxation / PLL, RG theory,
Hamiltonian mechanics. Much more directly useful — all three top
solutions converge on the same core architecture.

**Session 3** (`--prefer-categories "multiphysics, scientific machine
learning, statistical physics, fluid dynamics"`): gyrokinetic PIC,
variational integrators, Mori-Zwanzig projection, Floquet theory,
LES/turbulence closures, partitioned FSI, neural ODEs. The strongest
run — these communities have actually solved variants of our problem.
16 solutions across 8 domains.

---

## Key Insight: Three Independent Domains → Same Architecture

All three STEM-run top solutions (#1 Federated Kalman, #2 Polyphase
filterbank, #3 Waveform relaxation/PLL) independently recommend the
same four-component architecture:

1. **Waveform relaxation** at sync points (iterate until coupling
   converges, don't one-pass Lie split)
2. **IFT-based Jacobian** at sync boundaries (implicit function theorem
   for differentiation without unrolling inner iterations)
3. **Hermite interpolants** between sync points (continuous coupling,
   not frozen snapshots)
4. **Adaptive sync window** sizing based on coupling strength

The three solutions differ only in metaphor:
- Kalman: sync = "fusion epoch", coupling weight = "beta partition"
- Filterbank: sync = "analysis-synthesis window", convergence = "alias cancellation"
- PLL: sync = "PLL lock cycle", convergence = "Gauss-Seidel sweep"

Solution #3 (waveform relaxation in SPICE) is barely even an analogy —
it **is** the same problem (coupled circuit subcircuits at different
timescales).

---

## Consolidated Priority List

All three sessions' suggestions, deduplicated and ranked:

**Status column re-checked 2026-09-05.** Three rows read "Done" or as a
clean win and are contradicted by filed defects — shipped is not the same
as working, and this table was the thing asserting otherwise.

| # | What | Sessions that recommend it | Effort | Impact |
|---|------|--------------------------|--------|--------|
| 1 | Strang splitting (symmetric half-kick) | Gyrokinetics, Variational, FSI, design doc | Low | Shipped, **does not attain O(dt²)** — P0.2 |
| 2 | Interpolated coupling (dense output) | Mycorrhizal, all STEM, all multiphysics | Low | Shipped, **reaches only group gi-1** — P0.23 |
| 3 | Adaptive macro_dt (PLL-inspired) | PLL, Kalman, Filterbank, Gyrokinetics | Low | Shipped, **6-20x worse on a cyclic split** — P0.49 |
| 4 | Waveform relaxation (Gauss-Seidel at sync) | PLL, FSI, Variational, LES | Medium | **The remaining fix** — measured trigger in P0.47 |
| 5 | Memory kernel for fast→slow coupling | Mori-Zwanzig | Med-High | Novel — captures history |
| 6 | Anderson acceleration for waveform relax | Filterbank, Variational, FSI | Low (once #4 exists) | Superlinear convergence |
| 7 | Residual spectral monitoring | LES, Filterbank | Low | Diagnostic / early warning |
| 8 | IFT adjoint at sync boundaries | All solutions, DEQ literature | Medium | Required by #4 *if* its sweep count adapts — see #4 |
| 9 | IMEX schemes | Immunological, Kalman | High | Deferred: needs custom solver |

---

## Tier 1: Do Now

### 1. Strang Splitting (symmetric half-kick)

**Source**: Gyrokinetics #1 (Boris algorithm half-kick), Variational
integrators #2 (VPRK), FSI #7 (predictor-corrector), and HallSim's
own design doc (noted as "future option").

**The problem**: Lie splitting solves groups sequentially — group A
sees the old state of B, then B sees the new state of A. This is
first-order accurate: error ~ O(macro_dt).

**The idea**: Strang splitting (symmetric): solve A for half a step,
then B for a full step, then A for the remaining half step. This
cancels the leading-order commutator error, raising accuracy to
O(macro_dt²).

Three independent domains (gyrokinetics, variational integration, FSI)
arrived at this independently. The gyrokinetics version (Boris push)
is the most concrete: half-electric-kick → magnetic rotation →
half-electric-kick. The FSI version calls it predictor-corrector
symmetrization.

**Implementation**: `coupling_mode="strang"` in the scheduler. For
each macro step, solve each group for dt/2, then solve in reverse
order for dt/2. Straightforward modification of the Lie loop.

**Status**: IMPLEMENTED in `scheduler.py`, but **it does not attain the
second order this entry is claiming** — observed order decays 1.16 -> 0.63
(P0.2). On a cycle-spanning split it is 7x *worse* than Lie and non-monotone
(P0.47). Do not cite the O(dt^2) line as a property of the shipped code.

### 2. Dense Output Interpolation Coupling

**Source**: Mycorrhizal analogy (session 1), independently confirmed by
all three STEM solutions (session 2, step 4a in each).

**The problem**: Lie splitting freezes the slow group's state while the
fast group integrates.

**The idea**: Pass a continuous interpolant (cubic Hermite spline) from
the previous group's Diffrax solution. The next group's RHS queries
this interpolant at the current time `t`.

**Status**: IMPLEMENTED in `scheduler.py` as `coupling_mode="interpolated"`.
Demo: `demos/multiscale_coupling_demo.py` (39x error reduction on slow
coupling variable at macro_dt=2.0) — **but only when the driving edge is
adjacent in group order.** The interpolant is built for group `gi-1` alone, so
any earlier group stays frozen while the mode reports as interpolated (P0.23).
The 39x demo happens to be an adjacent configuration.

### 2. Adaptive macro_dt (PLL-inspired)

**Source**: PLL solution #3 (step 3), Kalman solution #1 (step 3),
Filterbank solution #2 (step 3). All three recommend it.

**The problem**: Fixed `macro_dt` is either too large (coupling drift →
solver explosion) or too small (unnecessary overhead). The right
`macro_dt` changes during the simulation as dynamics stiffen or relax.

**The idea**: After each macro step, measure the coupling residual
(how much the state changed between the start and end of the Lie
splitting pass). If the residual is growing, shrink `macro_dt`. If
it's been small for several consecutive steps, grow it.

Concretely (PLL lock-detector analogy):
- Compute `rho = ||state_after_split - state_before_split|| / ||state||`
- If `rho > rho_max` (e.g., 0.5): `macro_dt *= 0.5` (lose lock → shrink)
- If `rho < rho_min` (e.g., 0.01) for 3 consecutive steps:
  `macro_dt *= 2.0` (locked → grow)
- Clamp to `[macro_dt_min, macro_dt_max]`

**Why it matters**: This directly addresses the "solver explodes"
problem. The explosions likely come from `macro_dt` being too large for
a stiff coupling regime. Adaptive sizing catches it before it blows up.

**Theoretical backing**: All three solutions cite the optimal sync
window result: T_window* ~ C / sqrt(rho(J_coupling)), i.e., the
optimal window scales inversely with the square root of the coupling
Jacobian spectral radius. The adaptive controller approximates this
without needing to compute the Jacobian explicitly.

**Status**: IMPLEMENTED in `scheduler.py` — and **measured 6-20x worse than
fixed `macro_dt` on a cycle-spanning split** (P0.49), at 1.5x the wall clock.

The pseudocode above says why. `rho` is
`||state_after_split - state_before_split|| / ||state||`: it measures how much
the state **moved** over the window, not how much the split **erred**. When a
coupling cycle spans groups, the damage is a stale edge — one group reading a
macro-step-old value — and that does not announce itself as a large state
change. So `rho` stays under `rho_min`, the lock detector reads "locked", and
the controller *grows* the step in exactly the regime that needed it smaller.
A residual that could see this has to compare against a re-solve (i.e. one
waveform-relaxation sweep, #4), not against the window's own displacement.

### 3. Residual Spectral Monitoring (diagnostic)

**Source**: Filterbank solution #2 (prediction 1).

**The prediction**: Operator-splitting error produces a predictable
"temporal aliasing" signature — residual power concentrates at harmonics
of 1/macro_dt in the Fourier transform of the coupling trajectory.

**The idea**: FFT the inter-group coupling residual time series. If
power at harmonics of 1/macro_dt is growing, the splitting is failing
before the solver explodes. An early-warning diagnostic.

**Status**: Not implemented. Low effort (~20 lines), useful as a
diagnostic tool.

---

## Tier 2: Medium Effort, High Impact

### 4. Waveform Relaxation at Sync Points

**Source**: All three STEM solutions. PLL #3 is the most direct (it
IS waveform relaxation, not an analogy for it).

**The problem**: One-pass Lie splitting (solve A, then B) loses cross-
coupling. The second group sees A's final state but A never saw B's
contribution within the macro step.

**The idea**: After one Lie pass, check the coupling residual. If it's
above tolerance, re-solve the groups (Gauss-Seidel iteration) until
the coupling converges. Typically 2-3 passes suffice.

```
repeat:
    state_old = state.copy()
    for group in groups:
        state = solve_group(group, state, t, t_next)
    residual = ||state - state_old|| / ||state||
until residual < epsilon or k > k_max
```

**Enhancement — Anderson acceleration** (filterbank #2, step 4b):
Store last m=3-5 iterates, solve a small least-squares for optimal
mixing coefficients. Converges superlinearly vs linearly for plain
Gauss-Seidel. Standard technique with off-the-shelf implementations.

**Interesting prediction** (filterbank #2): The number of Anderson
iterations needed to converge = effective coupling rank of the
feedback loop at that time point. If it suddenly jumps mid-simulation,
something biologically interesting is happening (bifurcation).

**Implementation**: ~40 lines wrapping the existing Lie loop. Anderson
acceleration adds ~30 more.

**Status**: Not implemented. **Now has a measured trigger and cost
(P0.47, 2026-09-05).**

The generic statement above — "one-pass Lie loses cross-coupling" —
understates when it bites. Splitting is *fine* on an acyclic group
DAG: the driver runs first and the consumer reads a fresh value. It
fails when a **coupling cycle spans groups**, because then no
execution order exists in which both directions are fresh, and one
edge is always a macro-step stale. `Composite.cyclic_group_sets()`
reports exactly this condition, so the iteration can be *skipped*
where it buys nothing and spent only on the cyclic sets.

Measured on the three-model multi-hallmark composite (DP14 86400 s /
GZ06 3600 s / K14 60 s, cycle `damage_bridge → dp14 → gz06 →
p53_cdkn1a`), against the exact single-group solve:

| | macro_dt 3.5 | 1.75 | 1.0 | 0.5 |
|---|---|---|---|---|
| Lie, fast group first | 10.00% | 3.07% | 1.96% | 1.30% |
| Lie, slow group first | 14.03% | 2.85% | 1.58% | 0.77% |
| Strang | 72.0% | — | 29.2% | 8.8% |
| Lie + `adaptive_dt` | 65.2% | — | 38.6% | — |

Group order buys under 2× and flips sign with `macro_dt`; Strang is
worse here (P0.2); `adaptive_dt`, the one knob documented for this, is
6–20× worse and slower (P0.49). Only shrinking `macro_dt` helps, at
O(dt¹). So this is the fix, and there is currently no substitute.

**Two JAX constraints the pseudocode above does not survive.**

1. `until residual < epsilon or k > k_max` is a data-dependent trip
   count. Under `jit` that needs `lax.while_loop`, or a **static**
   sweep count with a fixed unroll. Prefer the static count: it keeps
   the traced shape constant, which is what lets the whole solve stay
   one compiled executable.
2. `lax.while_loop` is **not reverse-differentiable**, and Calibrator
   backprops through the solve. So a convergence-tested loop forecloses
   fitting unless #6 (IFT at sync boundaries) lands with it — which is
   the real reason those two entries belong together, not just an
   optimization. A static unroll differentiates fine and is the
   cheaper first step; add IFT when the sweep count needs to adapt.

Land it with P0.23's fix — both rewrite `_run_scan_continuous` and the
eager path, and doing them separately means paying that review twice.

### 5. Memory Kernel for Fast→Slow Coupling (Mori-Zwanzig)

**Source**: Mori-Zwanzig projection (#3, session 3). The most original
idea across all three sessions.

**The problem**: Even with interpolated coupling, the slow group sees
only the *current* fast-group state. But in biology, the slow response
depends on the *recent history* of the fast dynamics (e.g., Wnt
signaling history determines transcriptional response via pathway
desensitization, receptor recycling).

**The idea**: After each macro step, fit a short exponential-decay
memory kernel from the last N fast-group outputs. Feed the convolution
result to the slow group instead of the raw instantaneous output:

    M(t) = sum_k K(k*dt) * z_fast(t - k*dt)

If K decays within 1-2 macro steps → Markovian coupling is fine
(current approach works). If K has long tails → you need this memory
term or you're missing real physics.

**Practical version**: Don't implement full Mori-Zwanzig formalism.
Just: maintain a circular buffer of recent fast-group outputs, fit a
low-rank Volterra kernel via regularized least-squares, and feed the
convolution to the slow group. Re-estimate the kernel when the slow
state shifts substantially (detect via sliding-window KL divergence).

**Key prediction**: The kernel decay time tracks the fast sub-model's
autocorrelation time. If tau_K / tau_fast is consistently outside
[0.5, 2.0], the kernel is capturing slow-variable contamination, not
fast-mode relaxation — a diagnostic for bad scale separation.

**Status**: Not implemented. Novel for HallSim.

### 6. IMEX (Implicit-Explicit) Schemes

**Source**: Immunological analogy (session 1), referenced by Kalman
solution #1.

**The idea**: Additive Runge-Kutta schemes treating stiff terms
implicitly and coupling terms explicitly. Solve the coupled system in
one pass, not via splitting.

**Implementation complexity**: HIGH. Diffrax doesn't ship IMEX.

**Status**: Deferred — waveform relaxation + adaptive macro_dt is
likely sufficient and much easier.

---

## Tier 3: For When Optimization Works

### 6. IFT at Sync Boundaries

**Source**: All three STEM solutions (step 5 in each).

**The idea**: When doing adjoint-based parameter fitting, don't
backprop through waveform relaxation iterations. Use the implicit
function theorem: d(u*)/dθ = -[∂R/∂u*]^{-1} · [∂R/∂θ], computed
once at the converged fixed point. One sparse linear solve of size
n_ports × n_ports, not unrolling k inner iterations.

**Related work**: Deep Equilibrium Models (Bai et al., 2019) use
exactly this trick.

**Status**: Not implemented. Relevant when adjoint pipeline exists.

### 7. Per-Module Gradient Normalization

**Source**: PLL #3 (step 7), Kalman #1 (step 8).

**The prediction**: Gradient magnitude decays as rho^K across K sync
boundaries. For long simulations, slow-module parameters get vanishing
gradients. Fix: normalize adjoint per-module at each sync boundary
(the "DLL deskew buffer" analog).

**Status**: Theoretical. Relevant when optimization pipeline exists.

### 8. Hub Coupling Bottleneck Warning

**Source**: Immunological analogy (session 1).

**The prediction**: State variables read by many processes (e.g., ATP
in ERiQ) dominate gradient flow and starve peripheral modules.

**Status**: Theoretical. Monitor per-module adjoint norms when
optimization work begins.

### 9. Coarse-to-Fine Tolerance Schedule

**Source**: Immunological analogy (session 1).

**The idea**: Start optimization with loose solver tolerances, tighten
as parameters converge.

**Status**: Trivial callback. Low priority.

---

## Predictions Worth Testing

From all three sessions, deduplicated:

1. **Coupling bandwidth > solver accuracy** (mycorrhizal, all STEM):
   Halving `macro_dt` improves accuracy more than halving `rtol` for
   the same compute budget. Test with ERiQ.

2. **Optimal macro_dt is U-shaped** (PLL #3, filterbank #2): Too
   small = overhead; too large = coupling drift. The minimum cost is at
   macro_dt* ~ C/sqrt(spectral_radius(J_coupling)).

3. **Splitting error has spectral signature** (filterbank #2): FFT the
   coupling residual — error power at harmonics of 1/macro_dt.

4. **Mismatched primal-adjoint solvers → systematic gradient bias**
   (filterbank #2): Using different solvers for forward and backward
   passes introduces bias that doesn't shrink with tighter tolerance.
   Non-obvious, important for future adjoint work.

5. **Convergence bifurcation in tight SCCs** (PLL #3): Feedback loops
   with extreme timescale separation have a critical coupling strength
   below which Gauss-Seidel diverges regardless of tolerance — must
   switch to Newton. Sharp transition, not gradual.

6. **Strang splitting raises accuracy to O(dt²)** (gyrokinetics,
   variational, FSI): Symmetric half-step splitting cancels the
   leading-order commutator error. Testable by convergence study.

7. **Memory kernel decay ↔ fast autocorrelation** (Mori-Zwanzig):
   The fitted memory kernel's e-folding time should track the fast
   sub-model's autocorrelation time within a factor of 2.

8. **Conservation drift from omitting conservation residuals**
   (variational #2): Secular O(N_macro) drift in conserved quantities
   when only consistency (not conservation) is enforced at interfaces.

9. **Corrector iteration count as stability oracle** (gyrokinetics,
   FSI): Rising iteration count at sync points precedes visible
   trajectory error — free early warning diagnostic.

10. **Slow-manifold approximation failure is locally detectable**
    (gyrokinetics #1): Increasing variance in cycle-averaged fast
    output signals breakdown before slow-model error increases.

---

## Ideas Evaluated and Rejected

- **Module manifesto / interface contract**: We already have this
  (Process/Port/Topology).
- **Coupling graph construction**: We already have this (Composite +
  validation layer).
- **Treeverse checkpointing**: Standard adjoint technique. Premature.
- **CLE bridge for stochastic modules**: Premature — no stochastic
  processes exist yet.
- **Polyphase decomposition of black-box modules**: Filterbank #2
  suggested this but acknowledged it breaks for unknown transfer
  functions. Not applicable.
- **H-tree gradient aggregation**: PLL #3 noted this only works for
  acyclic coupling. ERiQ has feedback loops. Not applicable directly.

---

## SAPPhIRE Abstractions

### Session 1 (biology lenses)
**Information preservation across scale boundaries** — analogous to the
Renormalization Group consistency condition. Operator splitting loses
information, not just accuracy, at scale boundaries.

### Session 2 (STEM lenses)
**Adjoint symmetry under Legendre duality + adiabatic invariance** —
the variational principle that integrating out fast degrees of freedom
while preserving their causal imprint on slow manifolds requires
self-consistent renormalization of the effective coupling. Connects to:
Born-Oppenheimer approximation, Mori-Zwanzig projection, Pontryagin's
maximum principle. Gradient flow preservation = co-state analog of
Liouville's theorem.

---

## Physics Learning Resources

### Renormalization Group (accessible introductions)
- Sethna, "Statistical Mechanics: Entropy, Order Parameters, and Complexity" — Ch. 12, free PDF at sethna.lassp.cornell.edu, best starting point for non-physicists
- Kadanoff, "Statistical Physics: Statics, Dynamics and Renormalization" — Ch. 12-14
- Goldenfeld, "Lectures on Phase Transitions and the Renormalization Group" — standard graduate text

### Multi-Scale Numerical Methods
- Weinan E, "Principles of Multiscale Modeling" (Cambridge, 2011) — THE reference for HMM
- Kevrekidis et al., "Equation-free, coarse-grained multiscale computation" (Comm. Math. Sci., 2003)
- Engquist & Tsai, "Heterogeneous multiscale methods: A review" (Comm. Comp. Phys., 2009)

### Waveform Relaxation (the most directly applicable technique)
- Lelarasmee, Ruehli & Sangiovanni-Vincentelli, "The waveform relaxation method for time-domain analysis of large scale integrated circuits" (IEEE CAD, 1982) — the original paper
- Gander & Stuart, "Space-time continuous analysis of waveform relaxation for the heat equation" (SIAM J. Sci. Comp., 1998) — convergence theory
- Bartel & Günther, "A multirate W-method for electrical networks" (J. Comp. Appl. Math., 2002) — multirate extension

### SAPPhIRE Framework
- Chakrabarti et al., "A functional modeling approach..." (AIEDAM, 2005)
- SAPPhIRE = State-Action-Part-Phenomenon-Input-oRgan-Effect

### Operator Splitting
- McLachlan & Quispel, "Splitting methods" (Acta Numerica, 2002)
- Hairer, Lubich & Wanner, "Geometric Numerical Integration" — Ch. III

### IMEX Schemes
- Kennedy & Carpenter, "Additive Runge-Kutta schemes..." (Appl. Num. Math., 2003)
- Ascher, Ruuth & Spiteri, "Implicit-explicit Runge-Kutta methods..." (Appl. Num. Math., 1997)

### Federated Kalman Filtering
- Carlson, "Federated square root filter for decentralized parallel processors" (IEEE Trans. Aero. Elec. Sys., 1990)

### Implicit Differentiation Through Fixed Points
- Bai, Kolter & Koltun, "Deep Equilibrium Models" (NeurIPS 2019) — IFT for differentiating through iterative solvers

### Mori-Zwanzig Projection (session 3)
- Zwanzig, "Nonequilibrium Statistical Mechanics" (Oxford, 2001) — the textbook
- Chorin, Hald & Kupferman, "Optimal prediction and the Mori-Zwanzig representation of irreversible processes" (PNAS, 2000) — accessible introduction
- Parish & Duraisamy, "A paradigm for data-driven predictive modeling using field inversion and machine learning" (JCP, 2016) — learned closure models

### Gyrokinetic / PIC Methods (session 3)
- Boris, "Relativistic plasma simulation" (1970) — the Boris push
- Brizard & Hahm, "Foundations of nonlinear gyrokinetic theory" (Rev. Mod. Phys., 2007)

### Variational / Geometric Integration (session 3)
- Marsden & West, "Discrete mechanics and variational integrators" (Acta Numerica, 2001)
- Hairer, Lubich & Wanner, "Geometric Numerical Integration" (Springer, 2006)

### Partitioned FSI (session 3)
- Degroote, Bathe & Vierendeels, "Performance of a new partitioned procedure versus a monolithic procedure in fluid-structure interaction" (Comp. Struct., 2009)
- Küttler & Wall, "Fixed-point fluid-structure interaction solvers with dynamic relaxation" (Comp. Mech., 2008)

### LES / Turbulence Closures (session 3)
- Germano et al., "A dynamic subgrid-scale eddy viscosity model" (Phys. Fluids A, 1991) — the Germano identity
