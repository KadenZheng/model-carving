"""Muon: Nesterov momentum, Polar Express, NorMuon."""

import jax.numpy as jnp

POLAR = [  # Polar Express, 5 steps
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def orthogonalize(g):
    x = g.reshape(g.shape[0], -1).astype(jnp.float32)
    scale = jnp.sqrt(jnp.maximum(1.0, x.shape[1] / x.shape[0]))
    x = x / (jnp.linalg.norm(x) * 1.02 + 1e-6)
    tall = x.shape[0] >= x.shape[1]
    for a, b, c in POLAR:
        A = x.T @ x if tall else x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + (x @ B if tall else B @ x)
    return (x * scale).reshape(g.shape).astype(g.dtype)


def muon(g, m, v, momentum=0.95, beta2=0.9):
    """Muon direction with NorMuon row/col normalization."""
    m = momentum * m + (1 - momentum) * g
    o = orthogonalize((1 - momentum) * g + momentum * m)
    axis = -1 if o.shape[-2] >= o.shape[-1] else -2
    mean_sq = jnp.mean(o.astype(jnp.float32) ** 2, axis=axis, keepdims=True)
    v = (1 - beta2) * mean_sq + beta2 * v
    n = o.shape[axis]
    step = jnp.maximum(v, 1e-10) ** -0.5
    scale = step * (jnp.sqrt(jnp.sum(mean_sq * n)) / jnp.maximum(jnp.sqrt(jnp.sum((mean_sq * n) * step**2)), 1e-10))
    return (o * scale).astype(o.dtype), m, v
