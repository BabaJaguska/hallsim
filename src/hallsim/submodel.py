from abc import ABC, abstractmethod

# Global registry for all known submodels
# ⚠️ Populated automatically via the @register_submodel decorator at import time.
# Do NOT modify manually — instead, define your model and decorate it.
SUBMODEL_REGISTRY = {}


def register_submodel(name):
    """
    Decorator to register a submodel class under a given name.
    Enables plugin-style discovery and dynamic construction.
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
    def inputs(self) -> set[str]:
        pass

    @abstractmethod
    def outputs(self) -> set[str]:
        pass
