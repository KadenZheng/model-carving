"""Looped layers."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from carving.model.config import ModelConfig
from carving.model.layers import EPS, F32, attention, dense, inv_freq, rms_norm


class LoopedAttention(nn.Module):
    cfg: ModelConfig

    def setup(self):
        d, uniform = self.cfg.n_embd, nn.initializers.variance_scaling(1.0, "fan_in", "uniform")
        self.c_q, self.c_k, self.c_v = (dense(d, False, uniform, n) for n in ("c_q", "c_k", "c_v"))
        self.c_proj = dense(d, False, nn.initializers.zeros, "c_proj")

    def __call__(self, x, q_src=None):
        """q_src, the prev loop's output for queries from the second loop on."""
        B, T, C = x.shape
        H, D = self.cfg.n_head, C // self.cfg.n_head
        q = self.c_q(x if q_src is None else q_src).reshape(B, T, H, D)
        k, v = self.c_k(x).reshape(B, T, H, D), self.c_v(x).reshape(B, T, H, D)
        f = jnp.arange(T, dtype=jnp.int32)[None, :, None].astype(F32) * inv_freq(D)[None, None, :]
        cos, sin = jnp.cos(f)[:, :, None, :], jnp.sin(f)[:, :, None, :]

        def rotate(t):  # rotary enc, then QK-norm
            t1, t2 = t[..., : D // 2], t[..., D // 2 :]
            return rms_norm(jnp.concatenate([t1 * cos + t2 * sin, -t1 * sin + t2 * cos], -1), EPS)

        return self.c_proj(attention(rotate(q), rotate(k), v).reshape(B, T, C))


class LoopedMLP(nn.Module):
    cfg: ModelConfig

    def setup(self):
        d = self.cfg.n_embd
        self.c_fc = dense(4 * d, False, nn.initializers.variance_scaling(1.0, "fan_in", "uniform"), "c_fc")
        self.c_proj = dense(d, False, nn.initializers.zeros, "c_proj")

    def __call__(self, x):
        return self.c_proj(jax.nn.relu(self.c_fc(x)) ** 2)


class LoopedBlock(nn.Module):
    cfg: ModelConfig

    def setup(self):
        self.attn, self.mlp = LoopedAttention(self.cfg), LoopedMLP(self.cfg)

    def read_attn(self, z, q_src):
        return self.attn(rms_norm(z, EPS), q_src)

    def read_mlp(self, z):
        return self.mlp(rms_norm(z, EPS))
