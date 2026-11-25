"""Optimizer: Muon for matrices, AdamW for the rest."""

import jax
import jax.numpy as jnp
import optax

from carving.residual import update_connections
from carving.training.muon import muon


def labels(params):
    """Param groups: matrix, embedding, scalar, attn_input, conn."""
    def label(path, x):
        names = [str(getattr(k, "key", k)) for k in path]
        if names[0] == "conn":
            return "conn"
        if names[0] == "wte":
            return "embedding"
        return ("matrix" if x.ndim >= 2
                else "attn_input" if "ln_1" in names or names[-2:] == ["c_attn", "bias"] else "scalar")

    return jax.tree_util.tree_map_with_path(label, params)


def init_optimizer(params):
    lab = labels(params)
    mats = {path: x for path, x in _items(params) if _get(lab, path) == "matrix"}
    adam = {path: x for path, x in _items(params) if _get(lab, path) in ("embedding", "scalar", "attn_input")}
    return {
        "muon_m": {p: jnp.zeros_like(x) for p, x in mats.items()},
        "muon_v": {p: jnp.zeros((*x.shape[:-1], 1) if x.shape[-2] >= x.shape[-1] else (*x.shape[:-2], 1, x.shape[-1]),
                                jnp.float32)
                   for p, x in mats.items()},
        "embedding": optax.scale_by_adam(0.8, 0.95, 1e-10).init(
            {p: x for p, x in adam.items() if _get(lab, p) == "embedding"}),
        "scalar": optax.scale_by_adam(0.8, 0.95, 1e-10).init(
            {p: x for p, x in adam.items() if _get(lab, p) != "embedding"}),
        "conn_m": jnp.zeros_like(params["conn"]) if "conn" in params else None,
        "conn_v": jnp.zeros_like(params["conn"]) if "conn" in params else None,
        "conn_n": jnp.zeros(params["conn"].shape, jnp.int32) if "conn" in params else None,
    }


def optimizer_update(params, grads, opt, lr, tc, wd_mult, lr_mult):
    """Clips to global norm 1, then steps each group."""
    lab = labels(params)
    backbone = _clip({p: g for p, g in _items(grads) if _get(lab, p) != "conn"})
    new, opt = dict(_items(params)), dict(opt)
    m, v = dict(opt["muon_m"]), dict(opt["muon_v"])
    for p, g in backbone.items():
        if _get(lab, p) == "matrix":
            u, m[p], v[p] = muon(g, m[p], v[p])
            u = u + tc["weight_decay"] * wd_mult * new[p]
            new[p] = new[p] + (u * -lr) * lr_mult
    opt["muon_m"], opt["muon_v"] = m, v
    for group, mult, names in (("embedding", tc["embedding_lr"], ("embedding",)),
                               ("scalar", tc["scalar_lr"], ("scalar", "attn_input"))):
        gs = {p: g for p, g in backbone.items() if _get(lab, p) in names}
        if not gs:
            continue
        u, opt[group] = optax.scale_by_adam(0.8, 0.95, 1e-10).update(_clip(gs), opt[group])  # each group clipped too
        for p in gs:
            d = u[p]
            if _get(lab, p) == "attn_input":  # decoupled decay
                d = d + tc["attn_input_decay"] * wd_mult * new[p]
            new[p] = new[p] + (d * -(lr * mult)) * lr_mult
    if "conn" in params:
        new[("conn",)], opt["conn_m"], opt["conn_v"], opt["conn_n"] = update_connections(
            params["conn"], grads["conn"], opt["conn_m"], opt["conn_v"], opt["conn_n"], tc["conn_lr"] * lr_mult)
    out = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), [new[p] for p, _ in _items(params)])
    return out, opt


def _items(tree):
    return [(tuple(str(getattr(k, "key", k)) for k in path), x)
            for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _get(tree, path):
    for k in path:
        tree = tree[k]
    return tree


def _clip(g):
    norm = optax.global_norm(g)
    return jax.tree.map(lambda t: jax.lax.select(norm < 1.0, t, t / norm.astype(t.dtype)), g)
