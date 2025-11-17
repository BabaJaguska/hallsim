"""
Hallmark abstraction layer.
Provides explicit handles for biological hallmarks of aging.
"""

from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, Any
import jax.numpy as jnp


@dataclass
class Hallmark:
    """
    Represents a hallmark of aging as an explicit control handle.

    Attributes:
        name: Human-readable name (e.g., "Mitochondrial Dysfunction")
        handle: Normalized severity level, typically ∈ [0, 1]
                0 = healthy/optimal, 1 = severely impaired
        description: Brief description of the hallmark
        parameter_mappings: Dict mapping parameter names to callables
                           that transform the handle into parameter values.
                           Example: {"MITO_DMG_RATE_SA": lambda h: 1.0 + h * 2.0}
        state_associations: Set of CellState variables associated with this hallmark
    """

    name: str
    handle: float = 0.0
    description: str = ""
    parameter_mappings: Dict[str, Callable[[float], float]] = field(
        default_factory=dict
    )
    state_associations: set = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate hallmark handle is in reasonable range."""
        if not (0.0 <= self.handle <= 1.0):
            raise ValueError(
                f"Hallmark handle should be in [0, 1], got {self.handle} for {self.name}"
            )

    def intervene(self, delta: float) -> "Hallmark":
        """
        Modify the hallmark handle by delta.
        Returns a new Hallmark instance with updated handle.

        Args:
            delta: Change in handle value (can be negative)

        Returns:
            New Hallmark instance with updated handle (clipped to [0, 1])
        """
        new_handle = jnp.clip(self.handle + delta, 0.0, 1.0)
        return Hallmark(
            name=self.name,
            handle=float(new_handle),
            description=self.description,
            parameter_mappings=self.parameter_mappings,
            state_associations=self.state_associations,
            metadata=self.metadata,
        )

    def set_handle(self, value: float) -> "Hallmark":
        """
        Set the hallmark handle to a specific value.
        Returns a new Hallmark instance.

        Args:
            value: New handle value (will be clipped to [0, 1])

        Returns:
            New Hallmark instance with updated handle
        """
        new_handle = jnp.clip(value, 0.0, 1.0)
        return Hallmark(
            name=self.name,
            handle=float(new_handle),
            description=self.description,
            parameter_mappings=self.parameter_mappings,
            state_associations=self.state_associations,
            metadata=self.metadata,
        )

    def get_parameter_values(self) -> Dict[str, float]:
        """
        Compute parameter values based on current handle.

        Returns:
            Dictionary of parameter names to values
        """
        return {
            param_name: mapping_fn(self.handle)
            for param_name, mapping_fn in self.parameter_mappings.items()
        }

    def __repr__(self):
        return f"Hallmark(name='{self.name}', handle={self.handle:.3f})"


def hallmark_factory(
    name: str,
    handle: float = 0.0,
    description: str = "",
    parameter_mappings: Optional[Dict[str, Callable]] = None,
    state_associations: Optional[set] = None,
    **metadata,
) -> Hallmark:
    """
    Factory function to create a Hallmark instance.
    Convenient for initialization with metadata.
    """
    return Hallmark(
        name=name,
        handle=handle,
        description=description,
        parameter_mappings=parameter_mappings or {},
        state_associations=state_associations or set(),
        metadata=metadata,
    )


"""
# Example usage:
# mito_dysfunction = hallmark_factory(
#   name="Mitochondrial Dysfunction",
#  handle=0.3,
# description="Impairment in mitochondrial function",
# parameter_mappings={
#   "MITO_DMG_RATE_SA": lambda h: 1.0 + h * 2.0,
# "ATP_PROD_RATE_SA": lambda h: 1.0 - h * 0.5
# }
# """
