"""Checkpoints."""

import pickle

import jax
from jax.sharding import NamedSharding, PartitionSpec as P


def save(path, state, carver, t):
    host = jax.device_get(state)
    if jax.process_index() == 0:
        with open(path, "wb") as f:
            pickle.dump({"state": host, "carver": carver, "t": t}, f)


def load(path, mesh):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    return jax.device_put(ck["state"], NamedSharding(mesh, P())), ck["carver"], ck["t"]
