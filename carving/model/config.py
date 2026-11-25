"""Model config."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    family: str  # "standard" or "looped"
    n_layer: int
    n_embd: int
    n_head: int
    loops: int
    vocab_size: int
    seq_len: int
    carving: bool = False  # Model Carving's gated writes + connections
    f32_residual: bool = False
    remat: bool = False
