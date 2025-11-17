from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global registry for all known submodels
# ⚠️ Populated automatically via the @register_submodel decorator at import time.
# Do NOT modify manually — instead, define your model and decorate it.
SUBMODEL_REGISTRY = {}


def register_submodel(name):
    """
    Decorator to register a submodel class under a given name.
    Enables plugin-style discovery and dynamic construction.
    Registers a model when the module is imported.
    """

    def wrapper(cls):
        SUBMODEL_REGISTRY[name] = cls
        return cls

    return wrapper


class Submodel(ABC):
    """
    Abstract base class for biological submodels.
    Each submodel must define:
      - __call__: how the state evolves
      - inputs: variables it reads from CellState
      - outputs: variables it writes to CellState
    """

    @abstractmethod
    def __call__(self, t, state, args=None):
        pass

    @abstractmethod
    def outputs(self) -> set[str]:
        pass

    def update_parameters(self, **params):
        """
        Required for making hallmark handles effective.
        Update model parameters dynamically.
        Supports both dict-style and attribute-style params.
        """
        if isinstance(getattr(self, "params", None), dict):
            for key, value in params.items():
                if key in self.params:
                    logging.info(f"Updating parameter {key} to {value}")
                    self.params[key] = value
                else:
                    logging.warning(
                        f"Parameter {key} not found in model {self.model_name} parameters."
                    )
                    logging.info(
                        "Available parameters: "
                        + ", ".join(self.params.keys())
                    )
        else:
            for key, value in params.items():
                if hasattr(self.params, key):
                    logging.info(f"Updating parameter {key} to {value}")
                    setattr(self.params, key, value)
                else:
                    available = (
                        self.params.__dict__.keys()
                        if hasattr(self.params, "__dict__")
                        else dir(self.params)
                    )
                    logging.warning(
                        f"Parameter {key} not found in model {self.model_name} parameters."
                    )
                    logging.info(
                        "Available parameters: "
                        + ", ".join(map(str, available))
                    )
