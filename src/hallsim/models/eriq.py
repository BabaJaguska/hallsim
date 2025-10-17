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

    def outputs(self) -> set[str]:
        return {
            "mito_damage",
            "mito_function",
            "mito_enzymes",
            "glycolysis",
            "glycolytic_enzymes",
            "mTOR_activity",
            "mTOR_integrator_c",
            "p53_activity",
            "p53_integrator_c",
            "ROS_activity",
            "ROS_integrator_c",
        }

    def compute_observables(self, state, eps=1e-6):
        """
        Compute intermediate observables based on the current state and parameters.
        NOTE: These variables are computed on the fly, and are not evolved over time.
        But they do change, and their corresponding cell states will NOT reflect this
        unless we re-calculate them after ODE integration.
        Maybe there's a better way to handle this?
        """

        params = self.params
        # ATP-related
        ATP_mito = state["mito_function"]
        ATP_g = state["glycolysis"]
        ATP_total = ATP_mito + ATP_g

        # ROS node
        ROS_SA = params["ROS_SA"]
        ROS = 10 * ROS_SA * state["ROS_activity"]

        # PTEN node
        PTEN_SA = params["PTEN_SA"]
        PTEN = PTEN_SA * (1.0 / (ATP_mito + eps))

        # AKT node
        AKT_SA = params["AKT_SA"]
        AKT = AKT_SA * (PTEN + ROS / 5.0 + 0.1)

        # AMPK node
        AMPK_SA = params["AMPK_SA"]
        AMPK = AMPK_SA * (1.0 / (ATP_total + eps))

        # NAD and SIRT
        NAD_RATE_SA = params["NAD_RATE_SA"]
        NAD_ratio = NAD_RATE_SA * ATP_mito
        SIRT_SA = params["SIRT_SA"]
        SIRT = SIRT_SA * NAD_ratio

        # PGC1a
        PGC1a_SA = params["PGC1a_SA"]
        PGC1a = PGC1a_SA * (AMPK + 0.1 * SIRT)

        # mTOR
        MTOR_SA = params["MTOR_SA"]
        mTORs = AKT - 4 * AMPK
        mTOR_activity = state["mTOR_activity"]
        mTORa = mTOR_activity - 1.5 * mTORs
        mTOR = MTOR_SA * (mTORa + mTORs + 1.0)

        # NFkB
        NFKB_SA = params["NFKB_SA"]
        NFkB = NFKB_SA * (0.25 * ROS + AKT + 0.25 * mTOR)

        # P53
        P53a_SA = params["P53_SA"]
        P53_Base = params["P53_base"]
        P53_Act = params["P53_Act"]  # p53 activator; change between 0.4 and 4
        P53s = 0.3 * (P53_Base - AKT - NFkB + 0.5 * ROS) * P53_Act
        P53a = state["p53_activity"] - P53s
        P53 = P53a_SA * (P53a + P53s)

        # FOXO
        FOXO_SA = params["FOXO_SA"]
        FOXO = FOXO_SA * (1.0 / (AKT + eps))

        # Free radicals
        FREERAD_SA = params["FREERAD_SA"]
        mito_damage = state["mito_damage"]
        ros_integrator = state["ROS_integrator_c"]
        radical_driver = FREERAD_SA * (
            P53 + 0.2 * mito_damage + ros_integrator - 0.05 * FOXO
        )

        # Autophagy
        AUTOPHAGY_SA = params["AUTOPHAGY_SA"]
        AUTOPHAGY = (
            AUTOPHAGY_SA
            * 0.001
            * (1.0 / (mTOR + eps) + 0.5 * FOXO + ROS + P53)
        )

        # HIF
        HIF_SA = params["HIF_SA"]
        HIF = HIF_SA * AKT

        # Pyruvate
        PYR_SA = params["PYR_SA"]
        pyruvate = PYR_SA * state["glycolysis"] * 0.7

        # glycolysis driver
        r2 = PGC1a + pyruvate + P53 - 0.2 * HIF - 0.2 * NFkB

        # mitochondrial damage drive
        MITO_DMG_RATE_SA = params["MITO_DMG_RATE_SA"]
        MDR = params[
            "MDR"
        ]  # mito damage rate, vary between 1.5*1e-3 and 2.6*1e-3.
        MD = MITO_DMG_RATE_SA * (
            abs(ATP_mito + ROS) * MDR + 0.0001 * (ROS - 0.8)
        )

        # glycolysis driver
        GLU_SA = params["GLU_SA"]
        glucose_uptake = GLU_SA * (1.0 / (NFkB + eps))
        r3 = glucose_uptake + 0.01 * HIF + 0.01 * NAD_ratio

        return {
            "ATP_mito": ATP_mito,
            "ATP_g": ATP_g,
            "ATP_total": ATP_total,
            "ROS": ROS,
            "PTEN": PTEN,
            "AKT": AKT,
            "AMPK": AMPK,
            "NAD_ratio": NAD_ratio,
            "SIRT": SIRT,
            "PGC1a": PGC1a,
            "mTOR": mTOR,
            "NFkB": NFkB,
            "P53": P53,
            "P53s": P53s,
            "FOXO": FOXO,
            "radical_driver": radical_driver,
            "AUTOPHAGY": AUTOPHAGY,
            "HIF": HIF,
            "pyruvate": pyruvate,
            "r2": r2,
            "MD": MD,
            "r3": r3,
        }

    def __call__(self, _t, state, args=None):
        eps = 1e-6
        obs = self.compute_observables(state, eps)

        mito_damage = state["mito_damage"]
        mito_enzymes = state["mito_enzymes"]
        glycolytic_enzymes = state["glycolytic_enzymes"]
        mTOR_activity = state["mTOR_activity"]
        mTOR_integrator_c = state["mTOR_integrator_c"]
        p53_integrator_c = state["p53_integrator_c"]
        ROS_integrator_c = state["ROS_integrator_c"]
        p53_activity = state["p53_activity"]

        gain2 = 0.01  # speed of response; mitochnodria respond slower than glycolysis
        # original says 0.05, but that makes mitochnodria behave erratically
        ry = 0.0
        rx = 0.0
        k3 = 1.0

        # Dynamics
        MDAMAGE_SA = self.params["MDAMAGE_SA"]
        dMito_damage = MDAMAGE_SA * (obs["MD"] - obs["AUTOPHAGY"])
        dMito_function = (
            gain2 * (obs["r2"] + mito_enzymes) - 0.02 * obs["SIRT"]
        )
        dMito_enzymes = (
            -k3 * obs["ATP_mito"] - (obs["r2"] - mito_damage) * mito_enzymes
        )  # change of mito enzymes is proportional to mitochnodrial damage (?)

        GLYCOL_SA = self.params["GLYCOL_SA"]
        dGlycolysis = GLYCOL_SA * (
            0.25 * (obs["r3"] + glycolytic_enzymes) + 1.0 / (obs["SIRT"] + eps)
        )
        dGlycolytic_enzymes = -obs["ATP_g"] - obs["r3"] * glycolytic_enzymes

        dmTOR_integrator_c = -mTOR_activity - mTOR_integrator_c
        dmTOR_activity = 0.1 * (ry + mTOR_integrator_c)

        dp53_integrator_c = -p53_activity - p53_integrator_c
        dp53_activity = 0.1 * (rx + p53_integrator_c)

        dROS_integrator_c = -obs["ROS"] - ROS_integrator_c
        dROS_activity = 0.01 * obs["radical_driver"]

        return {
            "mito_damage": dMito_damage,
            "mito_function": dMito_function,
            "mito_enzymes": dMito_enzymes,
            "glycolysis": dGlycolysis,
            "glycolytic_enzymes": dGlycolytic_enzymes,
            "mTOR_activity": dmTOR_activity,
            "mTOR_integrator_c": dmTOR_integrator_c,
            "p53_activity": dp53_activity,
            "p53_integrator_c": dp53_integrator_c,
            "ROS_activity": dROS_activity,
            "ROS_integrator_c": dROS_integrator_c,
        }

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
