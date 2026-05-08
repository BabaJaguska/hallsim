"""ModelAnalyzer — automated numerical and structural validation of Composites.

Tier 1: Pure algorithmic checks (no LLM required)
    - Jacobian stiffness analysis at initial condition
    - Singularity / blow-up detection via short burst simulation
    - Domain violation detection (negative concentrations, NaN, Inf)
    - Fixed-point proximity and stability analysis
    - Conservation law detection
    - Derivative magnitude profiling

Tier 2: LLM-assisted biological review (optional, pluggable)
    - Extracts model equations, port metadata, and ontology annotations
    - Formats a structured prompt for biological plausibility review
    - Parses structured LLM response into actionable findings

Usage
-----
>>> from hallsim.model_analyzer import ModelAnalyzer
>>> from hallsim.models.eriq import build_eriq_composite
>>> comp = build_eriq_composite()
>>> analyzer = ModelAnalyzer(comp)
>>> report = analyzer.run_all()
>>> print(report)

For LLM-assisted review::

    report = analyzer.biological_review(llm_fn=my_llm_call)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from hallsim.composite import Composite

log = logging.getLogger(__name__)


# ── Finding dataclass ──────────────────────────────────────────────────


@dataclass
class Finding:
    """A single analysis finding.

    Attributes
    ----------
    severity:
        "error", "warning", or "info".
    check:
        Name of the check that produced this finding.
    message:
        Human-readable description.
    details:
        Optional structured data (variable names, values, etc.).
    """

    severity: str  # "error" | "warning" | "info"
    check: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        tag = self.severity.upper()
        return f"[{tag}] ({self.check}) {self.message}"


@dataclass
class AnalysisReport:
    """Collection of findings from model analysis.

    Attributes
    ----------
    findings:
        List of Finding objects.
    metadata:
        Summary statistics and raw analysis data.
    """

    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def __str__(self):
        lines = [
            f"ModelAnalyzer Report: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s), "
            f"{len(self.findings) - len(self.errors) - len(self.warnings)} info",
            "",
        ]
        for f in self.findings:
            lines.append(f"  {f}")
        return "\n".join(lines)


# ── ModelAnalyzer ──────────────────────────────────────────────────────


class ModelAnalyzer:
    """Automated model validation for HallSim Composites.

    Parameters
    ----------
    composite:
        The wired Composite to analyze.
    y0:
        Initial state.  If None, uses ``composite.initial_state()``.
    """

    def __init__(
        self,
        composite: Composite,
        y0: dict[str, jnp.ndarray] | None = None,
    ):
        self.composite = composite
        self._rhs_flat, self._paths = composite.build_rhs()
        self.y0 = y0 if y0 is not None else composite.initial_state()
        self._n = len(self._paths)

    def rhs(self, t: float, y_dict: dict) -> dict:
        """Dict-faced eval — flatten, call flat RHS, unflatten.

        Cost-sensitive paths (Diffrax, ``jax.jit``, ``jax.grad``) should
        call ``self._rhs_flat`` directly so they don't pay the
        flatten/unflatten roundtrip per call.
        """
        return self.composite.unflatten(
            self._rhs_flat(t, self.composite.flatten(y_dict, self._paths)),
            self._paths,
        )

    # ── Check 1: Jacobian & stiffness ──────────────────────────────────

    def check_jacobian(self, t: float = 0.0) -> list[Finding]:
        """Compute Jacobian at IC, analyze eigenvalues for stiffness.

        Checks:
        - Stiffness ratio (max/min eigenvalue magnitude)
        - Positive real eigenvalues (potential instability)
        - Condition number of the Jacobian
        """
        findings = []
        y_vec = self.composite.flatten(self.y0, self._paths)

        J = jax.jacobian(lambda v: self._rhs_flat(t, v))(y_vec)
        J_np = np.array(J)

        eigenvalues = np.linalg.eigvals(J_np)
        real_parts = eigenvalues.real
        magnitudes = np.abs(eigenvalues)

        # Store raw data
        eig_data = {
            "eigenvalues": eigenvalues.tolist(),
            "real_parts": real_parts.tolist(),
            "magnitudes": magnitudes.tolist(),
            "variables": self._paths,
        }

        # Stiffness ratio
        nonzero_mags = magnitudes[magnitudes > 1e-12]
        if len(nonzero_mags) >= 2:
            stiffness_ratio = float(nonzero_mags.max() / nonzero_mags.min())
            eig_data["stiffness_ratio"] = stiffness_ratio
            if stiffness_ratio > 1e4:
                findings.append(
                    Finding(
                        severity="warning",
                        check="jacobian",
                        message=(
                            f"High stiffness ratio: {stiffness_ratio:.0f}. "
                            f"Consider an implicit solver (Kvaerno5) or check for "
                            f"near-singular terms in the equations."
                        ),
                        details=eig_data,
                    )
                )
            elif stiffness_ratio > 100:
                findings.append(
                    Finding(
                        severity="info",
                        check="jacobian",
                        message=f"Moderate stiffness ratio: {stiffness_ratio:.1f}.",
                        details=eig_data,
                    )
                )
            else:
                findings.append(
                    Finding(
                        severity="info",
                        check="jacobian",
                        message=f"Low stiffness ratio: {stiffness_ratio:.1f}. Non-stiff system.",
                        details=eig_data,
                    )
                )

        # Positive real eigenvalues → potential instability
        pos_real = real_parts[real_parts > 1e-10]
        if len(pos_real) > 0:
            findings.append(
                Finding(
                    severity="warning",
                    check="jacobian",
                    message=(
                        f"{len(pos_real)} eigenvalue(s) with positive real part "
                        f"(max: {float(pos_real.max()):.4e}). "
                        f"System may be locally unstable at IC."
                    ),
                    details={"positive_eigenvalues": pos_real.tolist()},
                )
            )

        # Condition number
        cond = float(np.linalg.cond(J_np))
        if cond > 1e10:
            findings.append(
                Finding(
                    severity="warning",
                    check="jacobian",
                    message=f"Jacobian condition number: {cond:.2e} (near-singular).",
                    details={"condition_number": cond},
                )
            )

        return findings

    # ── Check 2: Derivative magnitudes at IC ───────────────────────────

    def check_derivative_magnitudes(self, t: float = 0.0) -> list[Finding]:
        """Check for extreme derivative values at the initial condition.

        Flags:
        - Very large derivatives (potential blow-up)
        - Very small derivatives across all variables (dead system)
        - Large magnitude spread (timescale separation)
        """
        findings = []
        dy = self.rhs(t, self.y0)

        magnitudes = {}
        for p in self._paths:
            magnitudes[p] = abs(float(dy.get(p, 0.0)))

        max_mag = max(magnitudes.values())
        nonzero = {k: v for k, v in magnitudes.items() if v > 1e-15}
        min_nonzero = min(nonzero.values()) if nonzero else 0.0

        details = {
            "derivatives": {p: float(dy.get(p, 0.0)) for p in self._paths}
        }

        # Check for blow-up risk
        if max_mag > 1e6:
            worst = max(magnitudes, key=magnitudes.get)
            findings.append(
                Finding(
                    severity="error",
                    check="derivative_magnitudes",
                    message=(
                        f"Extreme derivative at IC: |d({worst})/dt| = {max_mag:.2e}. "
                        f"Likely numerical instability."
                    ),
                    details=details,
                )
            )
        elif max_mag > 1e3:
            worst = max(magnitudes, key=magnitudes.get)
            findings.append(
                Finding(
                    severity="warning",
                    check="derivative_magnitudes",
                    message=(
                        f"Large derivative at IC: |d({worst})/dt| = {max_mag:.2e}. "
                        f"May require small initial step size."
                    ),
                    details=details,
                )
            )

        # Check for dead system
        if max_mag < 1e-15:
            findings.append(
                Finding(
                    severity="warning",
                    check="derivative_magnitudes",
                    message="All derivatives are zero at IC. System is at a fixed point.",
                    details=details,
                )
            )

        # Timescale separation
        if min_nonzero > 0 and max_mag / min_nonzero > 1e4:
            findings.append(
                Finding(
                    severity="info",
                    check="derivative_magnitudes",
                    message=(
                        f"Derivative magnitude spread: {max_mag / min_nonzero:.0f}x. "
                        f"Large timescale separation between state variables."
                    ),
                    details=details,
                )
            )

        return findings

    # ── Check 3: Short burst simulation ────────────────────────────────

    def check_short_burst(
        self,
        t_burst: float = 10.0,
        dt: float = 0.001,
        max_steps: int = 50_000,
    ) -> list[Finding]:
        """Simulate a short burst and check for NaN, Inf, or blow-up.

        Parameters
        ----------
        t_burst:
            Duration of the test burst.
        dt:
            Fixed step size for Euler integration (avoids solver issues).
        max_steps:
            Maximum number of steps.
        """
        findings = []
        state = {k: jnp.array(float(v)) for k, v in self.y0.items()}

        n_steps = min(int(t_burst / dt), max_steps)
        t = 0.0
        blow_up_var = None
        blow_up_t = None

        for step in range(n_steps):
            dy = self.rhs(t, state)
            new_state = {}
            for p in self._paths:
                val = float(state[p]) + dt * float(dy.get(p, 0.0))
                if np.isnan(val) or np.isinf(val):
                    blow_up_var = p
                    blow_up_t = t
                    break
                if abs(val) > 1e15:
                    blow_up_var = p
                    blow_up_t = t
                    break
                new_state[p] = jnp.array(val)

            if blow_up_var is not None:
                findings.append(
                    Finding(
                        severity="error",
                        check="short_burst",
                        message=(
                            f"Numerical blow-up at t={blow_up_t:.4f} in variable "
                            f"'{blow_up_var}'. Check for singularities (1/x terms) "
                            f"or unbounded growth in the equations."
                        ),
                        details={
                            "blow_up_variable": blow_up_var,
                            "blow_up_time": blow_up_t,
                            "state_at_blow_up": {
                                k: float(v) for k, v in state.items()
                            },
                        },
                    )
                )
                return findings

            state = new_state
            t += dt

        # Check final state for unreasonable values
        for p in self._paths:
            val = float(state[p])
            init_val = float(self.y0[p])
            if init_val != 0 and abs(val / (init_val + 1e-15)) > 1e6:
                findings.append(
                    Finding(
                        severity="warning",
                        check="short_burst",
                        message=(
                            f"'{p}' grew by factor {abs(val / (init_val + 1e-15)):.0e} "
                            f"in {t_burst} time units. Possible runaway dynamics."
                        ),
                        details={
                            "variable": p,
                            "initial": init_val,
                            "final": val,
                        },
                    )
                )

        if not findings:
            findings.append(
                Finding(
                    severity="info",
                    check="short_burst",
                    message=f"Short burst (t={t_burst}, dt={dt}) completed without issues.",
                )
            )

        return findings

    # ── Check 4: Domain violations ─────────────────────────────────────

    def check_domain_violations(self) -> list[Finding]:
        """Check if initial state has obviously invalid values.

        Flags negative values for variables whose port descriptions or
        ontology suggest they should be non-negative (concentrations,
        activities, etc.).
        """
        findings = []
        concentration_hints = {
            "uM",
            "mM",
            "nM",
            "mol",
            "molecules",
            "cells",
            "count",
        }

        for proc_name, proc in self.composite.processes.items():
            schema = proc.ports_schema()
            topo = self.composite.topology.get(proc_name, {})
            for port_name, port in schema.items():
                store_path = topo.get(port_name)
                if store_path is None or store_path not in self.y0:
                    continue
                val = float(self.y0[store_path])

                # Check units hint for concentration-like variables
                is_concentration = (
                    any(
                        hint in port.units.lower()
                        for hint in concentration_hints
                    )
                    if port.units
                    else False
                )

                if is_concentration and val < 0:
                    findings.append(
                        Finding(
                            severity="warning",
                            check="domain",
                            message=(
                                f"'{store_path}' has negative initial value ({val:.4f}) "
                                f"but units='{port.units}' suggest non-negative quantity."
                            ),
                            details={
                                "path": store_path,
                                "value": val,
                                "units": port.units,
                            },
                        )
                    )

        return findings

    # ── Check 5: Singularity probe ────────────────────────────────────

    def check_singularity_probe(self, t: float = 0.0) -> list[Finding]:
        """Probe for latent singularities by perturbing each state variable toward zero.

        For each state variable, evaluates the RHS at a perturbed state
        where that variable is pushed close to zero.  If the derivative
        magnitude explodes (compared to the IC), flags a potential singularity.

        This catches ``1/x`` terms that are benign at the IC but will
        blow up as the simulation evolves.
        """
        findings = []
        dy_ic = self.rhs(t, self.y0)
        ic_norm = sum(abs(float(v)) for v in dy_ic.values()) + 1e-10

        for p in self._paths:
            ic_val = float(self.y0[p])

            # Probe near zero (but not exactly zero to avoid NaN)
            probe_val = 0.01
            # Also probe near negative if IC is positive
            if ic_val > 0.1:
                perturbed = {
                    k: jnp.array(probe_val if k == p else float(v))
                    for k, v in self.y0.items()
                }
            elif ic_val < -0.1:
                # For negative-valued variables, probe toward zero from below
                perturbed = {
                    k: jnp.array(-probe_val if k == p else float(v))
                    for k, v in self.y0.items()
                }
            else:
                continue  # Already near zero, skip

            try:
                dy_probe = self.rhs(t, perturbed)
                probe_norm = sum(abs(float(v)) for v in dy_probe.values())

                # Check for NaN/Inf
                has_nan = any(
                    np.isnan(float(v)) or np.isinf(float(v))
                    for v in dy_probe.values()
                )
                if has_nan:
                    findings.append(
                        Finding(
                            severity="error",
                            check="singularity_probe",
                            message=(
                                f"NaN/Inf in derivatives when '{p}' → {probe_val}. "
                                f"Likely a 1/x singularity in the equations."
                            ),
                            details={"variable": p, "probe_value": probe_val},
                        )
                    )
                    continue

                # Check for blow-up (>100x increase in derivative norm)
                ratio = probe_norm / ic_norm
                if ratio > 100:
                    # Identify which derivatives blew up
                    blown = []
                    for k in self._paths:
                        ic_mag = abs(float(dy_ic.get(k, 0.0)))
                        probe_mag = abs(float(dy_probe.get(k, 0.0)))
                        if probe_mag > max(ic_mag * 50, 1.0):
                            blown.append((k, float(probe_mag)))

                    blown_str = ", ".join(
                        f"{k} ({m:.1e})" for k, m in blown[:3]
                    )
                    findings.append(
                        Finding(
                            severity="warning",
                            check="singularity_probe",
                            message=(
                                f"Derivative blow-up ({ratio:.0f}x) when '{p}' → {probe_val} "
                                f"(IC: {ic_val:.4f}). Affected: {blown_str}. "
                                f"Suggests a reciprocal (1/x) or division by this variable."
                            ),
                            details={
                                "variable": p,
                                "ic_value": ic_val,
                                "probe_value": probe_val,
                                "derivative_ratio": ratio,
                                "blown_up_derivatives": blown,
                            },
                        )
                    )
            except Exception:
                findings.append(
                    Finding(
                        severity="error",
                        check="singularity_probe",
                        message=f"RHS evaluation failed when '{p}' → {probe_val}.",
                        details={"variable": p, "probe_value": probe_val},
                    )
                )

        if not findings:
            findings.append(
                Finding(
                    severity="info",
                    check="singularity_probe",
                    message="No latent singularities detected.",
                )
            )

        return findings

    # ── Check 6: Fixed-point analysis ──────────────────────────────────

    def check_fixed_point_proximity(self, t: float = 0.0) -> list[Finding]:
        """Check how close the IC is to a fixed point (dy/dt ≈ 0).

        Reports the relative magnitude of derivatives compared to state values.
        """
        findings = []
        dy = self.rhs(t, self.y0)

        relative_rates = {}
        for p in self._paths:
            dy_val = abs(float(dy.get(p, 0.0)))
            y_val = abs(float(self.y0[p]))
            # Relative rate: |dy/dt| / (|y| + epsilon)
            relative_rates[p] = dy_val / (y_val + 1e-10)

        max_rel = max(relative_rates.values())
        _ = np.mean(list(relative_rates.values()))  # available for diagnostics

        if max_rel < 1e-6:
            findings.append(
                Finding(
                    severity="info",
                    check="fixed_point",
                    message=(
                        f"IC is very close to a fixed point "
                        f"(max relative rate: {max_rel:.2e})."
                    ),
                    details={"relative_rates": relative_rates},
                )
            )
        elif max_rel < 1e-2:
            findings.append(
                Finding(
                    severity="info",
                    check="fixed_point",
                    message=(
                        f"IC is near a quasi-steady state "
                        f"(max relative rate: {max_rel:.2e})."
                    ),
                    details={"relative_rates": relative_rates},
                )
            )
        else:
            fastest = max(relative_rates, key=relative_rates.get)
            findings.append(
                Finding(
                    severity="info",
                    check="fixed_point",
                    message=(
                        f"IC is not at a fixed point. Fastest-changing variable: "
                        f"'{fastest}' (relative rate: {relative_rates[fastest]:.2e})."
                    ),
                    details={"relative_rates": relative_rates},
                )
            )

        return findings

    # ── Check 6: Conservation law detection ────────────────────────────

    def check_conservation_laws(self, t: float = 0.0) -> list[Finding]:
        """Detect approximate conservation laws (linear combinations with ~0 derivative).

        Uses the Jacobian's left null space to find conserved quantities.
        """
        findings = []
        y_vec = self.composite.flatten(self.y0, self._paths)
        J = np.array(jax.jacobian(lambda v: self._rhs_flat(t, v))(y_vec))

        # Left null space: rows of V^T where singular values ≈ 0
        U, S, Vt = np.linalg.svd(J)
        tol = max(J.shape) * S[0] * np.finfo(float).eps * 100

        null_dims = np.sum(S < tol)
        if null_dims > 0:
            # Get the conserved combinations from right null space of J^T
            # (= left null space of J)
            Ut_null = U[:, -null_dims:]  # last columns of U
            for i in range(null_dims):
                coeffs = Ut_null[:, i]
                nonzero = [
                    (self._paths[j], float(coeffs[j]))
                    for j in range(len(coeffs))
                    if abs(coeffs[j]) > 0.01
                ]
                if nonzero:
                    terms = " + ".join(f"{c:.2f}*{p}" for p, c in nonzero)
                    findings.append(
                        Finding(
                            severity="info",
                            check="conservation",
                            message=f"Approximate conservation law: {terms} ≈ const.",
                            details={"coefficients": dict(nonzero)},
                        )
                    )

        if not findings:
            findings.append(
                Finding(
                    severity="info",
                    check="conservation",
                    message="No conservation laws detected.",
                )
            )

        return findings

    # ── Run all checks ─────────────────────────────────────────────────

    def run_all(self, t_burst: float = 10.0) -> AnalysisReport:
        """Run all algorithmic checks and return a combined report.

        Parameters
        ----------
        t_burst:
            Duration for the short-burst stability test.
        """
        report = AnalysisReport()

        checks = [
            (
                "derivative_magnitudes",
                lambda: self.check_derivative_magnitudes(),
            ),
            ("jacobian", lambda: self.check_jacobian()),
            ("short_burst", lambda: self.check_short_burst(t_burst=t_burst)),
            ("domain", lambda: self.check_domain_violations()),
            ("fixed_point", lambda: self.check_fixed_point_proximity()),
            ("conservation", lambda: self.check_conservation_laws()),
            ("singularity_probe", lambda: self.check_singularity_probe()),
        ]

        for name, check_fn in checks:
            try:
                report.findings.extend(check_fn())
            except Exception as e:
                report.findings.append(
                    Finding(
                        severity="error",
                        check=name,
                        message=f"Check failed with exception: {e}",
                    )
                )

        return report

    # ── Tier 2: LLM-assisted biological review ─────────────────────────

    def extract_model_summary(self) -> str:
        """Extract a structured text summary of the model for LLM review.

        Returns a prompt-ready string describing:
        - All processes, their ports, roles, units, descriptions, ontology
        - The topology (wiring)
        - Initial conditions
        - Derivative values at IC
        """
        lines = ["# Model Summary for Biological Review", ""]

        # Processes and ports
        lines.append("## Processes and Ports")
        for proc_name, proc in self.composite.processes.items():
            lines.append(f"\n### Process: {proc_name}")
            lines.append(f"Class: {type(proc).__name__}")
            schema = proc.ports_schema()
            for port_name, port in schema.items():
                onto = f", ontology={port.ontology}" if port.ontology else ""
                lines.append(
                    f"  - {port_name}: role={port.role.value}, "
                    f"default={port.default}, units='{port.units}'"
                    f"{onto}"
                )
                if port.description:
                    lines.append(f"    Description: {port.description}")

        # Topology
        lines.append("\n## Topology (Wiring)")
        for proc_name, mapping in self.composite.topology.items():
            for port_name, store_path in mapping.items():
                lines.append(f"  {proc_name}.{port_name} -> {store_path}")

        # Initial state
        lines.append("\n## Initial State")
        for p in self._paths:
            lines.append(f"  {p} = {float(self.y0[p]):.6f}")

        # Derivatives at IC
        dy = self.rhs(0.0, self.y0)
        lines.append("\n## Derivatives at t=0")
        for p in self._paths:
            lines.append(f"  d({p})/dt = {float(dy.get(p, 0.0)):.6e}")

        return "\n".join(lines)

    def build_review_prompt(self) -> str:
        """Build a structured LLM prompt for biological plausibility review.

        Returns a prompt string that can be sent to any LLM API.
        """
        summary = self.extract_model_summary()

        prompt = f"""{summary}

# Review Instructions

You are a systems biology expert reviewing an ODE model for biological
plausibility. For each process and equation, assess:

1. **Sign correctness**: Are activatory/inhibitory relationships in the
   correct direction? (e.g., does PTEN inhibit AKT, not activate it?)

2. **Functional form**: Are the mathematical forms appropriate?
   - Are Hill/Michaelis-Menten functions used where saturation is expected?
   - Are there raw reciprocals (1/x) that should be replaced?
   - Are Hill coefficients in reasonable ranges?

3. **Parameter plausibility**: Are default values in biologically
   reasonable ranges given the units?

4. **Missing interactions**: Are there well-known biological interactions
   that are absent from the model?

5. **Ontology consistency**: Do the ontology annotations match the
   described biology?

# Output Format

Return a JSON array of findings, each with:
- "severity": "error" | "warning" | "info"
- "check": "biological_review"
- "variable": the affected variable/process
- "message": description of the issue
- "suggestion": recommended fix

Example:
[
  {{
    "severity": "error",
    "check": "biological_review",
    "variable": "PTEN",
    "message": "PTEN is modeled as activated by low mitochondrial function, but biologically PTEN is inactivated by ROS (oxidation of Cys124).",
    "suggestion": "Replace PTEN = 1/mito_function with PTEN = K/(K + ROS) (inhibitory Hill of ROS)."
  }}
]
"""
        return prompt

    def biological_review(
        self,
        llm_fn: Callable[[str], str] | None = None,
    ) -> list[Finding]:
        """Run LLM-assisted biological plausibility review.

        Parameters
        ----------
        llm_fn:
            A callable that takes a prompt string and returns the LLM's
            response string.  Should return JSON matching the format
            described in the prompt.

            If None, returns an empty list and logs a message.

        Returns
        -------
        List of Finding objects from the LLM review.
        """
        if llm_fn is None:
            log.info(
                "No LLM function provided. Call with llm_fn=your_function "
                "to enable biological review. The function should accept a "
                "prompt string and return a response string."
            )
            return []

        import json

        prompt = self.build_review_prompt()
        response = llm_fn(prompt)

        # Parse JSON response
        findings = []
        try:
            # Try to extract JSON from the response
            # Handle case where LLM wraps JSON in markdown code blocks
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            items = json.loads(text)
            for item in items:
                findings.append(
                    Finding(
                        severity=item.get("severity", "info"),
                        check="biological_review",
                        message=item.get("message", ""),
                        details={
                            "variable": item.get("variable", ""),
                            "suggestion": item.get("suggestion", ""),
                        },
                    )
                )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            findings.append(
                Finding(
                    severity="warning",
                    check="biological_review",
                    message=f"Could not parse LLM response: {e}",
                    details={"raw_response": response[:500]},
                )
            )

        return findings
