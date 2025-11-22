from hallsim.submodel import Submodel, register_submodel
from sbmltoodejax.utils import load_biomodel


@register_submodel("biomodels_mapk")
class BioModels_MAPK(Submodel):
    """
    BioModels MAPK cascade model.
    Simulates the MAPK signaling pathway.

    Species aliases:
    MKKK / MKKK_P    ~ Raf (MAPKKK inactive/active)
    MKK / MKK_P / MKK_PP   ~ MEK (MAPKK 0/1/2 P)
    MAPK / MAPK_P / MAPK_PP ~ ERK (MAPK 0/1/2 P; MAPK_PP is active ERK)
    P stands for phosphorylated.
    MAPK = Mitogen Activated Protein Kinase
    Role: Transmits signals from receptors on the cell surface to the DNA in the nucleus.
    """

    def __init__(self):
        super().__init__()
        self.model_name = "BioModels_MAPK"
        self.model, self.y0, self.w0, self.c = load_biomodel(10)

    def outputs(self) -> set[str]:
        return {
            "MAPK_active",
            "MAPKK_active",
            "MAPKKK_active",
            "MAPK_PP",
            # there's more
        }

    def __call__(self, t, state, args=None):
        # Prepare initial conditions
        y = self.y0.copy()
        # Map CellState to model state variables
        # Assuming state is a dict-like object with species concentrations
        y_indexes = self.model.modelstepfunc.y_indexes
        for idx, species in y_indexes.items():
            if species in state:
                y[idx] = state[species]

        # self.model.set_initial_conditions(y, self.w0, self.c) #?

        # get RHS
        dydt = self.model.modelstepfunc.ratefunc(y, t, self.w0, self.c)

        out_dict = {}
        for species, idx in y_indexes.items():
            out_dict[species] = dydt[idx]

        return out_dict
