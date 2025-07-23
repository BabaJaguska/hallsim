from hallsim.submodel import Submodel, register_submodel
import json
import os


@register_submodel("eriq")
class ERiQ(Submodel):
    """
    ERiQ model ported from MATLAB.
    Simulates ROS, ATP stress, and mitochondrial dynamics.
    """

    def __init__(self, config_file: str = "configs/eriq_config.json"):
        super().__init__()
        self.config_file = config_file
        self.params = self.read_config()

    def read_config(self) -> dict:
        """
        Read the ERiQ configuration from a JSON file.
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
            raise FileNotFoundError(
                f"Configuration file {config_path} not found."
            )

    def inputs(self) -> set[str]:
        return {
            "mito_function",
            "p53",
            "ROS",
            "glycolysis",
            "glycolytic_enzymes",
            "ATP_total",
        }

    def outputs(self) -> set[str]:
        return {
            # Core mitochondrial and glycolytic states
            "mito_damage",
            "mito_function",
            "mito_enzymes",
            "glycolysis",
            "glycolytic_enzymes",
            # Feedback integrators
            "mTOR",
            "p53",
            "ROS",
            # Energy sensors and shared nodes
            "AMPK",
            "ATP_mito",
            "ATP_gly",
            "ATP_total",
            "PGC1a",
            "SIRT",
            "NAD_ratio",
            "AKT",
            "PTEN",
            "P53a",
            "P53s",
            "NFkB",
            "AUTOPHAGY",
            "FOXO",
            "radical_driver",
            "glucose_uptake",
            "pyruvate",
        }

    def __call__(self, t, state, args=None):
        """
        Compute deltas for relevant cell states at time t.
        """
        # dummy implementation
        deltas = {key: 0.01 for key in self.outputs()}
        return deltas

    def __repr__(self):
        return "ERiQ Submodel: Simulates ROS, ATP stress, and mitochondrial dynamics"

    def __str__(self):
        return "ERiQ Submodel - A model for simulating cellular stress and mitochondrial function"
