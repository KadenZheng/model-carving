"""Standard layers (GPT-2/3)."""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from carving.model.config import ModelConfig
from carving.model.layers import F32, LayerNorm, attention, dense, inv_freq


class StandardAttention(nn.Module):
    cfg: ModelConfig

    def setup(self):
        d = self.cfg.n_embd
        self.c_attn = dense(3 * d, True, nn.initializers.normal(0.02), "c_attn")
        self.c_proj = dense(d, True, nn.initializers.normal(0.02 / math.sqrt(2 * self.cfg.n_layer)), "c_proj")

    def __call__(self, x):
        B, T, C = x.shape
        H = self.cfg.n_head
        q, k, v = (t.reshape(B, T, H, C // H) for t in jnp.split(self.c_attn(x), 3, axis=2))
        f = jnp.arange(T, dtype=jnp.int32)[None, :, None].astype(F32) * inv_freq(C // H)[None, None, :]
        emb = jnp.concatenate((f, f), axis=-1)
        cos, sin = jnp.cos(emb)[:, :, None, :], jnp.sin(emb)[:, :, None, :]
        rot = lambda t: jnp.concatenate((-t[..., t.shape[-1] // 2 :], t[..., : t.shape[-1] // 2]), -1)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        return self.c_proj(attention(q, k, v).reshape(B, T, C))


class StandardMLP(nn.Module):
    cfg: ModelConfig

    def setup(self):
        d = self.cfg.n_embd
        self.c_fc = dense(4 * d, True, nn.initializers.normal(0.02), "c_fc")
        self.c_proj = dense(d, True, nn.initializers.normal(0.02 / math.sqrt(2 * self.cfg.n_layer)), "c_proj")

    def __call__(self, x):
        return self.c_proj(jax.nn.gelu(self.c_fc(x)))


class StandardBlock(nn.Module):
    cfg: ModelConfig

    def setup(self):
        self.ln_1, self.attn = LayerNorm(), StandardAttention(self.cfg)
        self.ln_2, self.mlp = LayerNorm(), StandardMLP(self.cfg)

    def read_attn(self, z, q_src):
        return self.attn(self.ln_1(z))

    def read_mlp(self, z):
        return self.mlp(self.ln_2(z))
