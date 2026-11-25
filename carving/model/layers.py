"""Normalization, dense layers, attn and rotary frequencies"""

import flax.linen as nn
import jax
import jax.numpy as jnp

BF16, F32 = jnp.bfloat16, jnp.float32
EPS = 2.0**-23


def rms_scale(x):
    x = x.astype(F32)
    return jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS)


def normalize(x):
    """N(x) = x / rms(x), fp32."""
    return x.astype(F32) / rms_scale(x)


def rms_norm(x, eps):
    """Param-free RMSNorm"""
    x32 = x.astype(F32)
    return (x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)).astype(x.dtype)


def dense(features, bias, init, name):
    return nn.Dense(features, use_bias=bias, kernel_init=init, param_dtype=F32, dtype=BF16, name=name)


def attention(q, k, v):
    """Causal attn in bf16"""
    dev = jax.devices()[0]
    impl = "cudnn" if dev.platform == "gpu" and any(n in dev.device_kind for n in ("H100", "H200", "B200")) else None
    return jax.nn.dot_product_attention(q.astype(BF16), k.astype(BF16), v.astype(BF16), is_causal=True,
                                        implementation=impl)


def inv_freq(head_dim):
    return 1.0 / (10000 ** (jnp.arange(0, head_dim, 2, dtype=F32) / head_dim))


class LayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        w = self.param("weight", nn.initializers.ones, (x.shape[-1],), F32)
        b = self.param("bias", nn.initializers.zeros, (x.shape[-1],), F32)
        x32 = x.astype(F32)
        xn = (x32 - jnp.mean(x32, -1, keepdims=True)) / jnp.sqrt(jnp.var(x32, -1, keepdims=True) + 1e-5)
        return w.astype(BF16) * xn.astype(BF16) + b.astype(BF16)
