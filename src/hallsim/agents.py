import equinox as eqx
import numpy as np
import json
import os


class CellState(eqx.Module):
    """
    Biological state of a cell, including various metabolic and signaling pathways.
    Changes in CellState variables need to be followed by exact changes in configuration files.
    """

    mito_damage: float
    mito_function: float
    mito_enzymes: float
    glycolysis: float
    glycolytic_enzymes: float
    mTOR_integrator_c: float
    mTOR: float
    p53_integrator_c: float
    p53: float
    ros_integrator_c: float
    ROS: float
    AMPK: float

    def __repr__(self):
        return (
            f"CellState(AMPK={self.AMPK:.3f}, p53={self.p53:.3f}, "
            f"ROS={self.ROS:.3f}, mTOR={self.mTOR:.3f})"
        )


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
        print(config_path)

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
