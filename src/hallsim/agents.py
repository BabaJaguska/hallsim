import numpy as np
import json
import os
from dataclasses import dataclass
from hallsim.submodel import SUBMODEL_REGISTRY
from hallsim.models import eriq  # noqa: F401
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import diffrax as dfx

jax.config.update("jax_enable_x64", False)
# Enable 64-bit precision if needed for stability in ODE solving


@dataclass
class CellState:
    """
    Biological state of a cell, including various metabolic and signaling pathways.
    Changes in CellState variables need to be followed by exact changes in configuration files.
    """

    # -- Energy and metabolism --
    mito_damage: float  # mitochondrial structural damage
    mito_function: float  # mitochondrial output efficiency
    mito_enzymes: float  # mitochondrial enzyme levels
    glycolysis: float  # glycolytic flux
    glycolytic_enzymes: float  # glycolytic enzyme levels
    ATP_mito: float  # ATP from mitochondria
    ATP_gly: float  # ATP from glycolysis
    ATP_total: float  # total ATP available
    glucose_uptake: float  # extracellular glucose availability
    pyruvate: float  # glycolysis output
    HIF: float  # hypoxia-inducible factor, response to low oxygen

    # -- Feedback integrators and core nodes --
    mTOR_integrator_c: float
    mTOR: float  # mechanistic target of rapamycin (growth signal)
    mTOR_activity: float  # mTOR activity level
    p53_integrator_c: float
    p53: float  # tumor suppressor protein, stress response
    p53_activity: float
    ROS_integrator_c: float
    ROS: float  # reactive oxygen species
    ROS_activity: float  # ROS activity level

    # -- Regulatory signals --
    AMPK: float  # cellular energy sensor
    PTEN: float  # PI3K/AKT pathway inhibitor
    AKT: float  # metabolic growth signal
    NAD_ratio: float  # NAD+/NADH balance
    SIRT: float  # NAD-dependent deacetylase
    FOXO: float  # stress-response transcription factor
    PGC1a: float  # mitochondrial biogenesis driver
    P53a: float  # active form of p53
    P53s: float  # suppressed p53
    NFkB: float  # inflammation/stress response

    # -- System-wide effects --
    AUTOPHAGY: float  # cleanup/recycling activity
    radical_driver: float  # free radical generator input

    def state_to_pytree(self):
        """
        Convert CellState to a PyTree-compatible structure for jax/diffrax.
        """
        return {
            field: jnp.asarray(getattr(self, field))
            for field in self.__dataclass_fields__
        }

    def pytree_to_state(self, pytree):
        """
        Update CellState from a PyTree-compatible structure.
        """
        for key, value in pytree.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"CellState has no attribute '{key}'")

    def broadcast_to_full_derivative(self, partial_deriv):
        """
        Given a partial derivative dictionary, broadcast it to a full derivative
        dictionary matching all CellState fields, filling missing fields with zeros.
        """
        full_deriv = {
            field: jnp.zeros_like(getattr(self, field))
            for field in self.__dataclass_fields__
        }
        for key, value in partial_deriv.items():
            if key in full_deriv:
                full_deriv[key] = value
            else:
                raise AttributeError(f"CellState has no attribute '{key}'")
        return full_deriv


class Cell:
    """
    An individual cell agent with spatial coordinates and internal biological state.
    """

    def __init__(self, coords=(0, 0), state_file="default_cell_config.json"):
        self.coords = np.array(coords, dtype=float)
        self.coord_names = ["x", "y"]
        self.state = self.init_cell_state(state_file)
        self.state_dev = self.state.state_to_pytree()
        self.zero_template = tree_map(jnp.zeros_like, self.state_dev)
        self.solver = dfx.Tsit5()
        self.models = {
            name: SUBMODEL_REGISTRY[name]() for name in SUBMODEL_REGISTRY
        }

    def init_cell_state(self, state_file):
        """
        Load cell state configuration from a JSON file.
        """
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )
        config_path = os.path.join(project_root, "configs", state_file)

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return CellState(**config)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file {config_path} not found."
            )

    def __repr__(self):
        coord_str = ", ".join(
            f"{n}={v:.1f}" for n, v in zip(self.coord_names, self.coords)
        )
        return f"Cell({coord_str})"

    def __str__(self):
        coord_str = ", ".join(
            f"{n}={v:.1f}" for n, v in zip(self.coord_names, self.coords)
        )
        return f"Cell at ({coord_str}) with {self.state}"

    def step(
        self,
        t0: float,
        t1: float,
        dt: float = 1.0,
        model_names: list[str] = None,
        keep_trajectory=False,
    ):
        """
        Executes one simulation step for a single cell.

        Args:
            t: current time step (float)
            model_names: list of submodel names to apply. If None, applies all available models.
        """
        if model_names is None:
            model_names = list(self.models.keys())
        model_fns = [self.models[name].__call__ for name in model_names]

        def rhs(t, y, args=None):
            parts = []
            for f in model_fns:
                partial = f(
                    t, y, args
                )  # must return dict[str, jnp.ndarray] = derivatives (not dt-scaled)
                # pad missing keys with zeros so shapes match y
                full = {
                    k: (
                        jnp.asarray(partial[k], dtype=y[k].dtype)
                        if k in partial
                        else self.zero_template[k]
                    )
                    for k in y
                }
                parts.append(full)
            return tree_map(lambda *xs: sum(xs), *parts)

        if keep_trajectory:
            saveat = dfx.SaveAt(ts=jnp.arange(t0, t1 + dt, dt))
        else:
            saveat = dfx.SaveAt(t1=True)

        term = dfx.ODETerm(rhs)
        sol = dfx.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=1e-3,  # initial step size guess
            y0=self.state_dev,
            args=None,
            # save at every integer time step
            saveat=saveat,
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
            max_steps=400_000,  # increase if needed for stiff problems
        )
        self.state_dev = tree_map(
            lambda a: jnp.squeeze(jnp.asarray(a)), sol.ys
        )
        self.state.pytree_to_state(
            self.state_dev
        )  # mirror back only when you actually need host values
        # otherwise keep everything on device


def apply_kick(state_dict, kick_dict):
    full = {
        k: (
            jnp.asarray(kick_dict[k], dtype=state_dict[k].dtype)
            if k in kick_dict
            else jnp.zeros_like(state_dict[k])
        )
        for k in state_dict
    }
    return tree_map(jnp.add, state_dict, full)
