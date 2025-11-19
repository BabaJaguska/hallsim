from hallsim.submodel import Submodel, register_submodel
import json
import os


@register_submodel("saturating_removal")
class SaturatingRemoval(Submodel):
    """
    Simple ODE model for damage (D) and repair machinery.
    Based on Uri Alon's saturating damage removal model.

    dD/dt = eta * Tau - beta * D / (K + D) + gaussian_noise
    # Tau is age, to denote difference in timescales of t and Tau

    Where:
    - eta: damage production rate
    - beta: maximum repair capacity
    - K: Michaelis-Menten constant for repair; concentration of D (e.g. senescent cells)
      at which half of the maximum removal rate is reached.
    """

    def __init__(
        self, config_file: str = "configs/saturating_removal_config.json"
    ):
        super().__init__()
        self.config_file = config_file
        self.params = self.read_config()
        self.model_name = "saturating_removal"

    def read_config(self) -> dict:
        """
        Read the damage repair configuration from a JSON file.
        """
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../")
        )
        config_path = os.path.join(project_root, self.config_file)

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            # Return default parameters if config not found
            return {
                "eta_damage_production_rate": 0.5,
                "beta_max_repair_capacity": 1.0,
                "K_SR": 0.1,
            }

    def outputs(self) -> set[str]:
        return {
            "damage_D",
        }

    def __call__(self, _t, state, args=None):
        """
        Compute derivatives for damage and repair machinery.
        """
        # Extract state variables
        D = state.get("damage_D", 0.0)

        # Extract parameters
        eta = self.params.get("eta_damage_production_rate", 0.5)
        beta = self.params.get("beta_max_repair_capacity", 1.0)
        K = self.params.get("K_SR", 0.1)

        # Compute derivatives
        tau_scale = 1.0 / (24.0 * 365)  # scale time to years if _t is in hours
        Tau = _t * tau_scale
        dD = eta * Tau - (beta * D) / (K + D)

        return {
            "damage_D": dD,
        }

    def __repr__(self):
        return (
            "DamageRepair Submodel: Simulates damage accumulation and repair"
        )

    def __str__(self):
        out = """
        DamageRepair Submodel
        Based on Uri Alon's saturating damage removal model.
        """
        return out
