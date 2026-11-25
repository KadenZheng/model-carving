"""FLOP costs, budget, projected end."""

import math


class Budget:
    """Training FLOPs = 3x fwd. Budget = the full-depth run's."""

    def __init__(self, units, mcfg, tokens_per_update, cfg):
        d, n_layer = mcfg.n_embd, mcfg.n_layer
        self.tokens = tokens_per_update
        self.sub = {"attn": 24 * d * d + 12 * d * mcfg.seq_len, "mlp": 48 * d * d}
        self.readout = 6 * mcfg.vocab_size * d
        # 9 d per valid connection, fwd + bwd
        n_conn = sum(1 + k + (n_layer if r else 0) for r in range(mcfg.loops) for k in range(n_layer))
        self.connections = 9 * d * n_conn
        self.flops = cfg.full_depth_tokens * self.per_token(units, connections=False)
        self.spent = 0.0
        self.rate_start, self.rate_window, self.rate_readings = cfg.rate_start, cfg.rate_window, cfg.rate_readings

    def per_token(self, units, connections=True):
        flops = self.readout + sum(self.sub[s] for (_, _, s) in units)
        return float(flops + (self.connections if connections else 0))

    def unit(self, sublayer):
        return self.sub[sublayer]

    def per_update(self, units):
        """Training FLOPs + write penalty (at the connections' rate)."""
        return self.tokens * self.per_token(units) + self.tokens * self.connections

    def charge(self, kept, read=None, floor=False):
        """One update, plus snapshot-read and floor fwd passes."""
        fwd = lambda units: self.per_token(units) / 3.0 * self.tokens
        reads = (fwd(read) if read is not None else 0.0) + (fwd(kept) if floor else 0.0)
        self.spent += self.tokens * self.per_token(kept) + reads + self.tokens * self.connections

    def exhausted(self):
        return self.spent >= self.flops

    def fraction(self):
        return self.spent / self.flops

    def end(self, step, per_update):
        return step + int(max(self.flops - self.spent, 0.0) // per_update)

    def projected_end(self, step, kept, frozen):
        """Last update + 1, snapshot passes amortized."""
        cost = self.per_update(kept)
        if not frozen and step >= self.rate_start:
            end = max(self.end(step, cost), step + 1)
            windows = math.log(end / max(step, 1)) / (self.rate_window / 2.0)
            fwd = self.per_token(kept) / 3.0 * self.tokens
            cost = cost + fwd * (self.rate_readings + 1) * windows / max(end - step, 1)
        return self.end(step, cost)
