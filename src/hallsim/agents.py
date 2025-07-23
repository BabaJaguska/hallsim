import numpy as np
import json
import os
from dataclasses import dataclass
from hallsim.submodel import SUBMODEL_REGISTRY, merge_state_updates
from hallsim.models import eriq  # noqa: F401


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

    def __repr__(self):
        return (
            f"CellState(AMPK={self.AMPK:.3f}, p53={self.p53:.3f}, "
            f"ROS={self.ROS:.3f}, mTOR={self.mTOR:.3f})"
        )

    def apply_updates(self, updates):
        """
        Apply a dictionary of updates (deltas) to the cell's state.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + value)
            else:
                raise AttributeError(f"CellState has no attribute '{key}'")


class Cell:
    """
    An individual cell agent with spatial coordinates and internal biological state.
    """

    def __init__(self, coords=(0, 0), state_file="default_cell_config.json"):
        self.coords = np.array(coords, dtype=float)
        self.coord_names = ["x", "y"]
        self.state = self.init_cell_state(state_file)

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

    def step(self, t: float, model_names: list[str] = None, method="add"):
        """
        Executes one simulation step for a single cell.

        Args:
            t: current time step (float)
            model_names: list of submodel names to apply (default: all registered)
            method: merge strategy for variable updates ("add" or "mean")
        """
        model_names = model_names or list(SUBMODEL_REGISTRY)
        updates = []

        # Parallelize?
        for name in model_names:
            model_cls = SUBMODEL_REGISTRY[name]
            model = model_cls()
            delta = model(t, self.state)
            updates.append(delta)
        merged = merge_state_updates(updates, method=method)
        self.state.apply_updates(merged)
