"""Carver: host-side state of a carved run."""

import math

import numpy as np

from carving.budget import Budget
from carving.gates import Gates
from carving.rate import Snapshots

SUBLAYERS = ("attn", "mlp")


def all_units(loops, n_layer):
    """Units u = (loop, layer, sublayer)."""
    return [(r, k, s) for r in range(loops) for k in range(n_layer) for s in SUBLAYERS]


class Carver:
    """Gates, pressure, removal, cooldown."""

    def __init__(self, cfg, mcfg, tokens_per_update):
        self.cfg = cfg
        self.R, self.K = mcfg.loops, mcfg.n_layer
        self.units = all_units(self.R, self.K)
        self.index = {u: i for i, u in enumerate(self.units)}
        self.budget = Budget(self.units, mcfg, tokens_per_update, cfg)
        self.t_plain = cfg.full_depth_tokens / tokens_per_update  # full-depth run's updates
        self.kept = frozenset(self.units)
        self.pending = []  # (removal update, units)
        self.gates = Gates(len(self.units), init=cfg.gate_init, upper=cfg.gate_upper, lr=cfg.gate_lr)
        self.rate = Snapshots(cfg, tokens_per_update)
        self.v_end = 0.0  # v(T)
        self.next_step, self.total_tokens, self.remaining_tokens = 0, 0.0, 0.0
        self.stopping = False  # cooldown starts next update
        self.frozen = False  # topology final (cooldown)
        self.decay = None  # (cooldown start, last update + 1)
        self.end = None
        self.events = []

    def before_step(self, t):
        kept, gates = self.kept, self.gate_values()
        pressures = self._pressures(t)
        reading = self.rate.reading(t, self.frozen)
        refresh = self.rate.refresh(t, self.frozen, gates, kept)
        return kept, gates, pressures, reading, refresh, self._lr_mult(t), self._wd_mult(t)

    def after_step(self, t, ce, gate_grad, ce_snapshot=None, ce_floor=None):
        """Budget, rate, gate step, removal, cooldown. Returns events."""
        self.budget.charge(self.kept, self.rate.read_graph() if ce_snapshot is not None else None, ce_floor is not None)
        if not self.frozen:
            self._forecast(t + 1)
        gate_grad = np.asarray(gate_grad, np.float64).sum(0)  # sum over micro-batches
        self.rate.observe(t, ce, ce_snapshot, ce_floor)
        self._refresh_v_end()
        self._gate_step(t, gate_grad)
        if self.stopping and not self.frozen:
            self.frozen = True
            self._remove(t, force=True)
            start, end = self._schedule(t + 1)
            end = ((end - 1) // 10) * 10 + 1  # end on the 10-update log grid
            self.decay, self.end = (min(start, end), end), end
            self.events.append({"step": t + 1, "event": "cooldown", "start": self.decay[0], "end": end})
        if not self.stopping and t + 1 >= self._cooldown_start(t + 1):
            self.stopping = True
        self._remove(t)
        if not self.frozen:
            self._forecast(t + 1)
            self._refresh_v_end()
        events, self.events = self.events, []
        return events

    def gate_values(self):
        """Gates as (loops, layers, 2), removed = 0."""
        return self.gates.values((self.R, self.K, 2))

    def upcoming(self, t):
        """Graphs of scheduled removals, to compile ahead."""
        closed, graphs, kept = self._pending(lambda a: a == 0.0), [], self.kept
        for at in sorted({at for at, _ in self.pending if at - self.cfg.removal_lead <= t + 1}):
            kept = kept - {u for a, us in self.pending if a == at for u in us if u in closed}
            graphs.append(kept)
        return graphs

    def done(self, t):
        return (self.end is not None and t + 1 >= self.end) or self.budget.exhausted()

    def metrics(self):
        return {"carving/kept": len(self.kept), "carving/v_end": self.v_end, "carving/exponent": self.rate.fit.a,
                "carving/spent": self.budget.fraction(),
                "carving/gate_min": self.gates.lowest([self.index[u] for u in self.kept])}

    def result(self):
        return {"kept": sorted(self.kept), "gates": self.gates.as_list()}

    def _lr_mult(self, t):
        if self.decay is None or t < self.decay[0]:
            return 1.0
        start, end = self.decay
        return 1.0 - min((t - start) / max(end - start, 1), 1.0)

    def _wd_mult(self, t):
        """Keeps lr x wd x updates at the full-depth run's."""
        end = self.end if self.end is not None else self.budget.projected_end(t, self.kept, self.frozen)
        return float(self.t_plain / max(end, 1))

    def _pressure(self, t):
        """Eq. 5: pi_u = v(T) ln(1 + rho phi_u / (1 - phi_u))."""
        c = self.cfg
        v = 0.0 if t < c.pressure_start or self.frozen or not self.rate.seen else self.v_end
        kept = [u for u in self.units if u in self.kept]
        held = self._pending(lambda a: a == 0.0)
        per_token = self.budget.per_token(self.kept - held)
        rho = 1.0 if self.total_tokens <= 0.0 else min(max(self.remaining_tokens / self.total_tokens, 0.0), 1.0)
        pi = []
        for u in kept:
            cost = self.budget.unit(u[2])
            phi = cost / (per_token + cost if u in held else per_token)  # closed unit: add its own share back
            pi.append(v * math.log1p(rho * phi / (1.0 - phi)))
        return kept, [self.index[u] for u in kept], np.asarray(pi, np.float64), v

    def _pressures(self, t):
        _, idx, pi, _ = self._pressure(t)
        out = np.zeros(len(self.units), np.float32)
        out[idx] = pi
        return out.reshape(self.R, self.K, 2)

    def _gate_step(self, t, gate_grad):
        """Gate step. A gate at zero schedules removal."""
        c = self.cfg
        kept, idx, pi, v = self._pressure(t)
        self.gates.step(idx, gate_grad.reshape(-1)[idx], pi, self._lr_mult(t), hold_open=self.frozen)
        if self.frozen:
            return
        back = self._pending(lambda a: a > 0.0)
        if back:  # reopened gates cancel removal
            self.pending = [(at, [u for u in us if u not in back]) for at, us in self.pending]
        waiting = self._pending()
        closing = [u for u in kept if self.gates.value(self.index[u]) == 0.0 and u not in waiting]
        if closing and v > 0.0:
            # removal at the interval's end + removal_lead
            at = -(-(t + 1) // c.removal_every) * c.removal_every + c.removal_lead
            self.pending.append((at, closing))
            self.events.append({"step": t + 1, "event": "gate closed", "units": closing, "removal_at": at})

    def _pending(self, test=lambda a: True):
        return {u for _, us in self.pending for u in us if test(self.gates.value(self.index[u]))}

    def _remove(self, t, force=False):
        keep = []
        for at, us in self.pending:
            if at > t + 1 and not force:
                keep.append((at, us))
            else:
                gone = {u for u in us if self.gates.value(self.index[u]) == 0.0}
                self.kept = self.kept - gone
                if gone:
                    self.events.append({"step": t + 1, "event": "removed", "units": sorted(gone)})
        self.pending = keep

    def _refresh_v_end(self):
        """v(T) at projected T."""
        if self.total_tokens <= 0.0 or self.next_step <= 0:
            self.v_end = 0.0
        else:
            self.v_end = max(self.rate.value(math.log(self.total_tokens)), 0.0)

    def _forecast(self, next_step):
        end = self.budget.projected_end(next_step, self.kept, self.frozen)
        self.next_step, self.total_tokens, self.remaining_tokens = (
            next_step, end * self.budget.tokens, (end - next_step) * self.budget.tokens)

    def _cooldown_start(self, next_step):
        end = self.budget.projected_end(next_step, self.kept, self.frozen)
        return end - int(round(self.cfg.cooldown * end))

    def _schedule(self, freeze_step):
        end = self.budget.end(freeze_step, self.budget.per_update(self.kept))
        return max(end - int(round(self.cfg.cooldown * end)), freeze_step), end
