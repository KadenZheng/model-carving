"""Gates: Adam + log-penalty step, clipped at zero."""

import math

import jax.numpy as jnp
import numpy as np


class Gates:
    """One gate per unit, host-side, fp32 steps."""

    def __init__(self, n, init, upper, lr, b1=0.9, b2=0.99, eps=1e-8, log_eps=0.05):
        self.a = np.full(n, init if upper is None else min(init, upper), np.float64)
        self.m = np.zeros(n, np.float64)
        self.v = np.zeros(n, np.float64)
        self.count = np.zeros(n, np.int64)
        self.upper, self.lr, self.b1, self.b2, self.eps = upper, lr, b1, b2, eps
        self.k = math.log(1.0 / log_eps)  # ln 20: full pressure per 20x fall

    def step(self, idx, grad, pressure, lr_mult, hold_open=False):
        """Eq. 8 at idx. Clips at 0, or 1e-6 if hold_open."""
        n = len(self.a)
        g, pi, fresh = np.zeros(n), np.zeros(n), np.zeros(n, bool)
        g[idx], pi[idx], fresh[idx] = grad, pressure, True
        slope = pi / (self.k * np.maximum(self.a, 1e-12))

        a = jnp.asarray(self.a, jnp.float32)
        m, v = jnp.asarray(self.m, jnp.float32), jnp.asarray(self.v, jnp.float32)
        fresh_ = jnp.asarray(fresh)
        count = jnp.where(fresh_, jnp.asarray(self.count, jnp.int32) + 1, jnp.asarray(self.count, jnp.int32))
        gj = jnp.asarray(g, jnp.float32)

        m = jnp.where(fresh_, self.b1 * m + (1.0 - self.b1) * gj, m)
        v = jnp.where(fresh_, self.b2 * v + (1.0 - self.b2) * gj * gj, v)

        steps = jnp.maximum(count, 1).astype(jnp.float32)
        denom = jnp.sqrt(v / (1.0 - self.b2**steps)) + self.eps
        lr = jnp.float32(self.lr * lr_mult)
        new = a - lr * (m / (1.0 - self.b1**steps)) / denom
        new = jnp.maximum(new - lr * jnp.asarray(slope, jnp.float32) / denom, 0.0)
        if self.upper is not None:
            new = jnp.minimum(new, self.upper)
        new = jnp.where(fresh_, new, a)

        self.a, self.m, self.v = (np.asarray(x, np.float64) for x in (new, m, v))
        self.count = np.asarray(count, np.int64)
        if hold_open:
            self.a[idx] = np.where(self.a[idx] <= 0.0, 1e-6, self.a[idx])

    def values(self, shape):
        return self.a.astype(np.float32).reshape(shape)

    def value(self, i):
        return self.a[i]

    def lowest(self, idx):
        return float(self.a[idx].min())

    def as_list(self):
        return self.a.tolist()
