import jax
import jax.numpy as jnp
import equinox as eqx
from hallsim.submodel import Submodel, register_submodel
from typing import Dict
import os
import json


class Func(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, input_size, width_size, depth, *, key):
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=input_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.softplus,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.out_scale * self.mlp(y)


@register_submodel("neuralode")
class NeuralODESubModel(Submodel):
    func: Func
    fields: list
    model_name: str = "NeuralODE"

    def __init__(
        self,
        fields=None,
        width_size=64,
        depth=2,
        key=jax.random.PRNGKey(0),
        load_weights=True,
        weights_path=None,
        config_file=None,
    ):
        if fields is None:
            # Default to a small subset of cell state
            fields = ["test_field1", "test_field2"]
        self.fields = fields
        self.key = key
        self.config_file = config_file
        self.params = self.read_config()
        self.func = Func(len(fields), width_size, depth, key=key)
        if load_weights:
            if weights_path is None:
                weights_path = "trained_neuralode.pkl"
            if os.path.exists(weights_path):
                self.load_weights(weights_path)
            else:
                print(
                    f"Warning: Weights file {weights_path} not found. Using randomly initialized weights."
                )

    def read_config(self):
        if self.config_file is None:
            return {}
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

    def __call__(
        self, t: float, state: Dict[str, float], args=None
    ) -> Dict[str, float]:

        # Extract the relevant fields from the state
        y_vec = jnp.array([state[field] for field in self.fields])
        dy_vec = self.func(t, y_vec, args)
        # Map the output back to dictionary format
        derivs = {field: dy for field, dy in zip(self.fields, dy_vec)}
        return derivs

    def outputs(self):
        output_set = set(self.fields)
        return output_set

    def load_weights(self, weights_path):
        self.func = eqx.tree_deserialise_leaves(weights_path, self.func)
