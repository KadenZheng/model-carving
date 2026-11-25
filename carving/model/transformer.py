"""Transformer + next-token loss."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from carving.model.config import ModelConfig
from carving.model.layers import BF16, EPS, F32, LayerNorm, rms_norm
from carving.model.looped import LoopedBlock
from carving.model.standard import StandardBlock
from carving.residual import Connections, rescale, write


class Transformer(nn.Module):
    cfg: ModelConfig
    kept: frozenset = None  # executed units (loop, layer, "attn" | "mlp"), None = all

    def setup(self):
        c = self.cfg
        self.wte = self.param("wte", nn.initializers.normal(0.02), (c.vocab_size, c.n_embd), F32)
        block = StandardBlock if c.family == "standard" else LoopedBlock
        self.blocks = [block(c) for _ in range(c.n_layer)]
        if c.carving:  # c[r, k, s]: source s into layer k of loop r
            self.conn = self.param("conn", nn.initializers.zeros, (c.loops, c.n_layer, 1 + 2 * c.n_layer), F32)
        if c.family == "standard":
            self.ln_f = LayerNorm()

    @nn.nowrap
    def layer(self, z, k, q_src, g_attn, g_mlp, use_attn, use_mlp):
        blk, carving = self.blocks[k], self.cfg.carving
        if use_attn:
            z = write(z, blk.read_attn(z, q_src), g_attn, carving)
        if use_mlp:
            z = write(z, blk.read_mlp(z), g_mlp, carving)
        return z

    @nn.nowrap
    def loop(self, r, z, q_src, prev, gates, pressures, penalty):
        c, init = self.cfg, self.is_initializing()
        layer = Transformer.layer if (init or not c.remat) else _REMAT_LAYER
        if c.carving:
            mix = Connections(self.conn[r], c.n_layer, z, prev)
        outputs = []
        for k in range(c.n_layer):
            use = [init or self.kept is None or (r, k, s) in self.kept for s in ("attn", "mlp")]
            g = (gates[r, k, 0], gates[r, k, 1]) if c.carving and not init else (1.0, 1.0)
            z = layer(self, z, k, q_src, g[0], g[1], use[0], use[1])
            outputs.append(z)
            if c.carving:
                z = mix(z, k)
                if pressures is not None and not init:
                    penalty = penalty + mix.write_penalty(k, [(g[i], pressures[r, k, i]) for i in range(2) if use[i]])
                z = rescale(z)
        return z, outputs, penalty

    def __call__(self, idx, targets, gates=None, pressures=None):
        """Mean CE + write penalty. Gates, pressures: (loops, n_layer, 2)."""
        c, init = self.cfg, self.is_initializing()
        penalty = jnp.float32(0.0)
        x = jnp.take(self.wte, idx, axis=0).astype(BF16)
        h, prev = None, None
        for r in range(1 if init else c.loops):
            z = rms_norm(x, EPS) if c.family == "looped" else x  # each loop restarts from embeddings
            z = z.astype(F32) if c.f32_residual else z
            z, prev, penalty = self.loop(r, z, h, prev, gates, pressures, penalty)
            h = rms_norm(z, EPS) if c.family == "looped" else z  # next loop's q_src
        h = h.astype(BF16)
        if c.family == "standard":
            h = self.ln_f(h)
        logits = jnp.einsum("bth,vh->btv", h, self.wte.astype(BF16))
        if c.family == "looped":
            logits = 15.0 * jnp.tanh(logits.astype(F32) / 15.0)
        logp = jax.nn.log_softmax(logits.astype(F32).reshape(-1, logits.shape[-1]), axis=-1)
        ce = -jnp.sum(jnp.take_along_axis(logp, targets.reshape(-1)[:, None], axis=-1)[:, 0]) / targets.size
        return ce, penalty


_REMAT_LAYER = nn.remat(
    Transformer.layer, static_argnums=(2, 6, 7), policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    prevent_cse=True,
)
