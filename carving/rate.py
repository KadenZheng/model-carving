"""Rate of learning v(D) = -dL/d ln D (Eq. 4): snapshot readings, power-law fit."""

import math


class RateFit:
    """ln v = b - a ln D (Eq. 7), with a prior on a (Eq. 9)."""

    def __init__(self, prior, fit_start, min_count=4):
        self.prior, self.fit_start, self.min_count = prior, fit_start, min_count
        self.x, self.y, self.t = [], [], []
        self.weight = None  # prior's weight
        self.a, self.b = prior, None

    def add(self, x, r, t):
        """Adds reading r > 0 at ln-token x, update t, then refits."""
        self.x.append(x), self.y.append(math.log(r)), self.t.append(t)
        win = [i for i, ti in enumerate(self.t) if ti >= self.fit_start]
        if self.weight is None and len(win) >= self.min_count and self._span(win) >= 1.0:
            self.weight = self._sums(win)[2]  # = data's weight at the first fit
        if self.weight is None:  # slope at prior
            xb, yb, _, _ = self._sums(range(len(self.x)))
            self.a = self.prior
        else:
            xb, yb, sxx, sxy = self._sums(win)
            self.a = max((-sxy + self.weight * self.prior) / (sxx + self.weight), 0.0)
        self.b = yb + self.a * xb

    def value(self, ln_d):
        return 0.0 if self.b is None else math.exp(self.b - self.a * ln_d)

    def _span(self, idx):
        xs = [self.x[i] for i in idx]
        return max(xs) - min(xs)

    def _sums(self, idx):
        xs, ys = [self.x[i] for i in idx], [self.y[i] for i in idx]
        xb, yb = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
        sxx = math.fsum((x - xb) ** 2 for x in xs)
        sxy = math.fsum((x - xb) * (y - yb) for x, y in zip(xs, ys))
        return xb, yb, sxx, sxy


class Snapshots:
    """Two snapshots refreshed in turn, each read at 1/4 to 1/2 e-fold old."""

    def __init__(self, cfg, tokens_per_update):
        self.cfg, self.tps = cfg, tokens_per_update
        self.slot_step, self.slot_gates = [-1, -1], [None, None]
        self.slot_kept, self.slot_reads = [None, None], [[], []]
        self.fit = RateFit(cfg.rate_prior, cfg.rate_fit_start)
        self.floor = 0.0  # loss gap at zero lag
        self.seen = False
        self._read = None  # (update, graph) read this update

    def reading(self, t, frozen):
        """(slot, gates, graph) read at update t, or None."""
        self._read = None
        if frozen:
            return None
        ready = [k for k in (0, 1) if self.slot_step[k] >= 1 and t in self.slot_reads[k]]
        if not ready:
            return None
        k = max(ready, key=lambda k: math.log(t / self.slot_step[k]))
        self._read = (self.slot_step[k], self.slot_kept[k])
        return k, self.slot_gates[k], self.slot_kept[k]

    def refresh(self, t, frozen, gates, kept):
        """Slot to refresh at update t, or -1."""
        w, s = self.cfg.rate_window, self.slot_step
        if t < self.cfg.rate_start or frozen:
            return -1
        if min(s) < 0:
            slot = s.index(-1) if max(s) < 0 or math.log(t / max(s)) >= w / 2 else -1
        else:
            older = 0 if s[0] <= s[1] else 1
            slot = older if math.log(t / s[older]) >= w else -1
        if slot >= 0:
            lo, hi = t * math.exp(w / 2), t * math.exp(w)
            n = self.cfg.rate_readings
            self.slot_step[slot], self.slot_gates[slot], self.slot_kept[slot] = t, gates, kept
            self.slot_reads[slot] = sorted(
                {p for p in (int(math.ceil(lo + i * (hi - lo) / n)) for i in range(n)) if p < hi})
        return slot

    def read_graph(self):
        return self._read[1]

    def observe(self, t, ce, ce_snapshot, ce_floor):
        """Records the floor and the reading (Eq. 6)."""
        if ce_floor is not None:
            self.floor = ce_floor - ce
        if ce_snapshot is None:
            return
        d_s, d_t = self._read[0] * self.tps, t * self.tps
        r = (ce_snapshot - ce) / math.log(d_t / d_s)
        if ce_snapshot - ce > abs(self.floor):  # below the floor: not a reading
            self.fit.add(0.5 * (math.log(d_s) + math.log(d_t)), r, t)
        self.seen = True

    def value(self, ln_d):
        return self.fit.value(ln_d)
