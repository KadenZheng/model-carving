"""Training loop."""

import argparse
import itertools
import json
import math
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from carving.carver import Carver
from carving.model.config import ModelConfig
from carving.training.checkpoint import load, save
from carving.training.data import Data
from carving.training.step import arg_specs, init_state, make_fns


def run(cfg, carving, out=None, save_every=0, resume=None):
    tc, cc = cfg["train"], cfg["carving"]
    tc["conn_lr"] = cc["conn_lr"]
    mcfg = ModelConfig(**{**cfg["model"], **(cc.get("model", {}) if carving else {}), "carving": carving})
    out = Path(out or tc["out"] + ("_carved" if carving else "_full"))
    main_proc = jax.process_index() == 0
    if main_proc:
        out.mkdir(parents=True, exist_ok=True)

    mesh = Mesh(np.array(jax.devices()), ("data",))  # data parallel
    data = Data(Path(tc["data"]), tc["batch_size"], tc["micro"], mcfg.seq_len, tc["val_sequences"],
                NamedSharding(mesh, P(None, "data", None)))
    state = init_state(mcfg, mesh, tc["seed"])
    carving_cfg = argparse.Namespace(**cc, full_depth_tokens=tc["tokens"], cooldown=tc["cooldown"])
    carver = Carver(carving_cfg, mcfg, data.update_tokens) if carving else None
    total = int(tc["tokens"] // data.update_tokens)  # full-depth run's updates
    decay_start = total - int(total * tc["cooldown"])
    lr0, t0 = tc["lr"], 0
    if resume:
        state, carver, t0 = load(resume, mesh)
        if (carver.done(t0 - 1) if carver is not None else t0 >= total):  # already complete
            return
    fns, log = {}, open(out / "metrics.jsonl", "a") if main_proc else None
    pool = ThreadPoolExecutor(1)  # background compile of the next graph
    specs = arg_specs(mcfg, state, data.batch(0), data.val) if carving else None

    def prepare(kept):
        if kept not in fns:
            fns[kept] = pool.submit(make_fns, mcfg, kept, tc, data.shard, specs)
        return fns[kept]

    def compiled(kept):
        return prepare(kept).result()

    def record(row):
        if main_proc:
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()

    def validate(t, kept, gates):
        nll = float(compiled(kept)[2](state["params"], data.val, gates)) / data.val_tokens
        record({"step": t, "val/loss": nll, "val/bpb": nll * data.val_tokens / (math.log(2) * data.val_bytes)})

    tic = time.time()
    for t in itertools.count(t0):
        batch = data.batch(t)
        if carver is None:  # full depth: constant lr, linear cooldown
            kept, gates, pressures, reading, refresh, lr_mult, wd_mult = None, None, None, None, -1, 1.0, 1.0
            lr = lr0 if t < decay_start else lr0 * (1.0 - (t - decay_start) / (total - decay_start))
        else:
            kept, gates, pressures, reading, refresh, lr_mult, wd_mult = carver.before_step(t)
            lr = lr0
        step, score, _ = compiled(kept)
        ce_snap = ce_floor = None
        if reading is not None:  # snapshot loss on this batch
            slot, s_gates, s_kept = reading
            ce_snap = compiled(s_kept)[1](state["snap"][slot], batch, s_gates)
        if refresh >= 0:  # numerical floor
            ce_floor = score(state["params"], batch, gates)
        state, ce, gate_grads = step(state, batch, gates, pressures, lr, wd_mult, lr_mult, refresh)
        ce = float(ce)
        if carver is not None:
            for event in carver.after_step(t, ce, np.asarray(jax.device_get(gate_grads)),
                                           None if ce_snap is None else float(ce_snap),
                                           None if ce_floor is None else float(ce_floor)):
                record(event)
            for graph in carver.upcoming(t):  # compile ahead of the switch
                prepare(graph)
        done = (carver.done(t) and t % 10 == 0) if carver is not None else t + 1 >= total
        if t % 10 == 0 or done:
            row = {"step": t, "train/loss": ce, "train/tokens": (t + 1) * data.update_tokens,
                   "time": round(time.time() - tic, 1)}
            record({**row, **carver.metrics()} if carver is not None else row)
        if t % tc["val_every"] == 0 or done:
            validate(t, carver.kept if carver else None, carver.gate_values() if carver else None)
        if save_every and (t + 1) % save_every == 0:
            save(out / "checkpoint.pkl", state, carver, t + 1)
        if done:
            break

    pool.shutdown(wait=False, cancel_futures=True)
    params = jax.device_get(state["params"])
    if main_proc:
        pickle.dump(params, open(out / "final_params.pkl", "wb"))
    if save_every:
        save(out / "checkpoint.pkl", state, carver, t + 1)
    if carver is not None and main_proc:
        json.dump({**carver.result(), "steps": t + 1}, open(out / "carved.json", "w"))
