"""Gated write + learned connections."""

import math

import jax
import jax.numpy as jnp

from carving.model.layers import F32, normalize, rms_scale


def write(z, y, g, carving):
    """z + y, or z + g rms(z) N(y) with carving (Eq. 1)."""
    if not carving:
        return z + y
    return z + (g * (rms_scale(z) * normalize(y))).astype(z.dtype)


def rescale(z):
    """Scales by the power of 2 nearest 1 / rms(z)."""
    e = jax.lax.stop_gradient(jnp.round(jnp.log2(rms_scale(z))))
    return (z.astype(F32) * jnp.exp2(-e)).astype(z.dtype)


class Connections:
    """u <- u + rms(u) sum_s c_ks (N(x_s) - N(u)) after layer k (Eq. 2)."""

    def __init__(self, coeff, K, loop_input, prev):
        self.a = coeff * jnp.asarray(valid_sources(K, prev is not None), F32)
        sources, cols = [normalize(loop_input)], self.a[:, :1]
        if prev is not None:
            sources += [normalize(p) for p in prev]
            cols = jnp.concatenate([cols, self.a[:, 1 + K :]], axis=1)
        stack = jnp.stack(sources)
        self.fixed = jnp.einsum("ks,s...->k...", cols, stack)  # loop input + prev loop, per layer
        self.fixed_sg = jnp.einsum("ks,s...->k...", cols, jax.lax.stop_gradient(stack))
        self.earlier, self.outputs = [], []  # N(u_j), u_j of earlier layers

    def __call__(self, u, k):
        nu = normalize(u)
        m = self.fixed[k]
        for j in range(k):
            m = m + self.a[k, 1 + j] * self.earlier[j]
        self.earlier.append(nu)
        self.outputs.append(u)
        return u + (rms_scale(u) * (m - jnp.sum(self.a[k]) * nu)).astype(u.dtype)

    def write_penalty(self, k, gate_pressures):
        """Gates' log penalty on connection-amplified writes, grad to c only."""
        sg = jax.lax.stop_gradient
        u = sg(self.outputs[k])
        nu = normalize(u)
        m = self.fixed_sg[k]
        for j in range(k):
            m = m + self.a[k, 1 + j] * sg(self.earlier[j])
        h = u.astype(F32) + rms_scale(u) * (m - jnp.sum(self.a[k]) * nu)
        ratio = jnp.sqrt(jnp.mean(u.astype(F32) ** 2, -1)) / jnp.sqrt(jnp.maximum(jnp.mean(h**2, -1), 1e-16))
        q = jnp.maximum((1.0 - jnp.sum(self.a[k])) * jnp.mean(ratio), 1.0)
        pen = jnp.float32(0.0)
        for g, pi in gate_pressures:
            g = sg(g)
            pen = pen + jnp.where(g > 0.0, pi / math.log(20.0) * jnp.log(jnp.maximum(g * q, 1e-30)), 0.0)
        return pen


def valid_sources(K, has_prev):
    """(K, 1 + 2K) mask: loop input, earlier layers, prev loop."""
    return [[1.0 if (s == 0 or s <= k or (K < s and has_prev)) else 0.0 for s in range(1 + 2 * K)] for k in range(K)]


def initial_connections(cfg):
    """Rows sum to 0.5."""
    rows = jnp.asarray([valid_sources(cfg.n_layer, r > 0) for r in range(cfg.loops)], F32)
    return rows * (0.5 / rows.sum(axis=-1, keepdims=True))


def update_connections(a, g, m, v, n, lr):
    """Adam, then row bounds. Zeros stay zero."""
    fresh = a != 0.0
    n = jnp.where(fresh, n + 1, n)
    m = jnp.where(fresh, 0.9 * m + (1.0 - 0.9) * g, m)
    v = jnp.where(fresh, 0.99 * v + (1.0 - 0.99) * g * g, v)

    k = jnp.maximum(n, 1).astype(jnp.float32)
    c = jnp.where(fresh, a - lr * (m / (1.0 - 0.9**k)) / (jnp.sqrt(v / (1.0 - 0.99**k)) + 1e-8), a)
    c = c * jnp.minimum(1.0, 1.0 / jnp.maximum(jnp.sum(jnp.abs(c), -1, keepdims=True), 1e-12))  # L1 norm <= 1
    c = c * (0.95 / jnp.maximum(jnp.sum(c, -1, keepdims=True), 0.95))  # signed sum <= 0.95
    return c, m, v, n
