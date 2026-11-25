"""Training state + compiled fns per graph."""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from carving.model.transformer import Transformer
from carving.residual import initial_connections
from carving.training.optimizer import init_optimizer, optimizer_update


def init_state(mcfg, mesh, seed):
    """Params + opt state. Carving adds connections and 2 snapshots."""
    def init(key):
        dummy = jnp.ones((1, mcfg.seq_len), jnp.int32)
        params = Transformer(mcfg).init(key, dummy, dummy)["params"]
        if mcfg.carving:
            params = {**params, "conn": initial_connections(mcfg)}
        state = {"params": params, "opt": init_optimizer(params)}
        if mcfg.carving:
            state["snap"] = [jax.tree.map(jnp.zeros_like, params) for _ in (0, 1)]
        return state

    return jax.jit(init, out_shardings=NamedSharding(mesh, P()))(jax.random.PRNGKey(seed))


def arg_specs(mcfg, state, batch, val):
    """Arg shapes, for compiling ahead of use."""
    spec = lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
    return (*jax.tree.map(spec, (state, batch, val)), jax.ShapeDtypeStruct((mcfg.loops, mcfg.n_layer, 2), jnp.float32))


def make_fns(mcfg, kept, tc, shard, specs=None):
    """Step, snapshot score, validation for one graph, compiled now if `specs`."""
    model = Transformer(mcfg, kept)
    carving = mcfg.carving

    def step(state, batch, gates, pressures, lr, wd_mult, lr_mult, refresh):
        params, acc = state["params"], batch[0].shape[0]

        def loss(p, g, x, y):
            ce, pen = model.apply({"params": p}, x, y, g, pressures)
            return ce / acc + pen / acc, ce / acc

        def body(carry, mb):
            gsum, lsum = carry
            if carving:
                (_, ce), (gp, gg) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(params, gates, *mb)
            else:
                (_, ce), gp = jax.value_and_grad(loss, has_aux=True)(params, gates, *mb)
                gg = jnp.zeros(())
            return (jax.tree.map(jnp.add, gsum, gp), lsum + ce), gg

        zero = jax.tree.map(jnp.zeros_like, params)
        (grads, ce), gate_grads = jax.lax.scan(body, (zero, jnp.float32(0.0)), batch)
        new, opt = optimizer_update(params, grads, state["opt"], lr, tc, wd_mult, lr_mult)
        out = {"params": new, "opt": opt}
        if carving:  # slot `refresh` takes the starting params
            out["snap"] = [jax.tree.map(lambda s, p: jnp.where(refresh == i, p, s), state["snap"][i], params)
                           for i in (0, 1)]
        return out, ce, gate_grads

    def score(params, batch, gates):  # batch loss, for snapshot readings
        acc = batch[0].shape[0]
        total, _ = jax.lax.scan(lambda s, mb: (s + model.apply({"params": params}, mb[0], mb[1], gates)[0] / acc, None),
                                jnp.float32(0.0), batch)
        return total

    def evaluate(params, batch, gates):  # summed val loss
        ces, _ = jax.lax.map(lambda mb: model.apply({"params": params}, mb[0], mb[1], gates), batch)
        return jnp.sum(ces * batch[0].shape[1] * batch[0].shape[2])

    rep = NamedSharding(shard.mesh, P())  # state replicated, batches split
    fns = (
        jax.jit(step, in_shardings=(rep, shard) + (None,) * 6, out_shardings=(rep, rep, rep), donate_argnums=0),
        jax.jit(score, in_shardings=(rep, shard, None)),
        jax.jit(evaluate, in_shardings=(rep, shard, None)),
    )
    if specs is None:
        return fns
    state, batch, val, gates = specs
    return (
        fns[0].lower(state, batch, gates, gates, 0.0, 0.0, 0.0, 0).compile(),  # pressures shaped like gates
        fns[1].lower(state["params"], batch, gates).compile(),
        fns[2].lower(state["params"], val, gates).compile(),
    )
