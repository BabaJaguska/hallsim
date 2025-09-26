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
            "mito_damage",
            "mito_function",
            "mito_enzymes",
            "glycolysis",
            "glycolytic_enzymes",
            "mTOR",
            "p53",
            "ROS",
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

    def __call__(self, _t, state, args=None):
        rate = 0.1
        return {k: rate * state[k] for k in self.outputs() if k in state}

    def __repr__(self):
        return "ERiQ Submodel: Simulates ROS, ATP stress, and mitochondrial dynamics"

    def __str__(self):
        out = """
        ERiQ Submodel - A model for simulating cellular stress and mitochondrial function.
        Adapted from the MATLAB implementation provided in
        [Alfego, D., & Kriete, A. (2017).
        Simulation of cellular energy restriction in quiescence (ERiQ)—a theoretical model for aging.
        Biology, 6(4), 44.]
        """
        return out
