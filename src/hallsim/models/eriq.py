from hallsim.submodel import Submodel, register_submodel


@register_submodel("eriq")
class ERiQ(Submodel):
    """
    ERiQ model ported from MATLAB.
    Simulates ROS, ATP stress, and mitochondrial dynamics.
    """

    # Maybe add parameters here later
    # my_param: float = eqx.field()

    def inputs(self) -> set[str]:
        return {
            "mito_function",
            "p53",
            "ROS",
            "glycolysis",
            "glycolytic_enzymes",
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
            "mTOR_integrator_c",
            "p53",
            "p53_integrator_c",
            "ROS",
            "ros_integrator_c",
            # Energy sensors and shared nodes
            "AMPK",
            "ATPm",
            "ATPg",
            "ATPr",
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
