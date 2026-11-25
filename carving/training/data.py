"""Token data."""

import jax
import numpy as np


def sequences(tokens, first, n, seq_len):
    """n sequences from token `first` + next-token targets, wrapping."""
    chunk = tokens[(first + np.arange(n * seq_len + 1)) % len(tokens)].astype(np.int32)
    return chunk[:-1].reshape(n, seq_len), chunk[1:].reshape(n, seq_len)


class Data:
    """Train batches + val window."""

    def __init__(self, folder, batch_size, micro, seq_len, val_sequences, shard):
        self.batch_size, self.micro, self.seq_len, self.shard = batch_size, micro, seq_len, shard
        self.update_tokens = batch_size * seq_len
        self.train = np.memmap(folder / "train.bin", dtype=np.uint32, mode="r")
        x, y = sequences(np.memmap(folder / "val.bin", dtype=np.uint32, mode="r"), 0, val_sequences, seq_len)
        self.val_bytes = int(np.load(folder / "byte_lengths.npy").astype(np.int64)[y].sum())  # UTF-8 bytes of targets
        self.val_tokens = x.size
        self.val = (self._global(x), self._global(y))

    def batch(self, t):
        x, y = sequences(self.train, t * self.update_tokens, self.batch_size, self.seq_len)
        return self._global(x), self._global(y)

    def _global(self, a):
        """Global array of micro-batches, split across devices."""
        a = a.reshape(-1, self.micro, self.seq_len)
        return jax.make_array_from_callback(a.shape, self.shard, lambda idx: a[idx])
