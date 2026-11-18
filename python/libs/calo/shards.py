"""Columnar shard storage (replaces ra_pickles and the DeepJetCore .djctd files).

The layout follows the fcc-ai-design shards (lyra/data/pf/<shard>/<array>.npy): a shard is one directory holding one
.npy file per column, plus offset arrays that cut the concatenated columns into records (simulations or events), and a
meta.json. Readers memory-map the columns, so a record is read without loading the shard, and the files are plain
numpy: no pickles, no container needed to read them.

  <shard>/meta.json            n records, the groups and their offset arrays, writer info and any user metadata
  <shard>/<group>_off.npy      int64 [n + 1] offsets of each variable-length group (e.g. hit_off for the hit_* columns)
  <shard>/<column>.npy         variable-length columns concatenated over records; per-record columns of length n

Grouping is by column-name prefix: every column whose name starts with a group's prefix shares that group's offsets
(all columns of a group must have the same length within a record). Columns matching no prefix hold one value
(scalar or fixed-shape array) per record. String values must be the same for every record of a shard (they go into
meta.json). Shards are written to <shard>.tmp and renamed, so a shard directory is always complete.
"""
import json
import os
import shutil
import socket
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

FORMAT = "hgcal-toy-shard-v1"

# simulation records (minicalo.simulate_* output); group name -> column prefix
SIM_GROUPS = {"rechit": "rechit_", "hit": "hit_particle_", "particle": "particles_"}
# event records (create_dataset.py output)
EVENT_GROUPS = {"hit": ""}          # every column of an event shard is per hit except those listed as per-event


def _group_of(key, groups):
    best = None
    for g, pre in groups.items():
        if key.startswith(pre) and (best is None or len(pre) > len(groups[best])):
            best = g
    return best


def write_shard(path, records, groups, meta=None, per_record=()):
    """Write a list of records (dicts of arrays / scalars) as one shard directory.

    groups: {group name: column prefix}; a column is variable-length if its name has one of the prefixes (the longest
    match wins) and is not listed in per_record. Returns the shard path."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        _write_columns(tmp, records, groups, meta, per_record)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)            # no half-written shard is left behind
        raise
    if path.exists():
        shutil.rmtree(path)
    os.rename(tmp, path)
    return path


def _write_columns(tmp, records, groups, meta, per_record):
    n = len(records)
    keys = list(records[0].keys()) if n else []
    for r in records[1:]:
        if set(r.keys()) != set(keys):
            raise ValueError(f"records of one shard must have the same keys: {sorted(set(r) ^ set(keys))}")
    constants = {}
    var_cols = {}
    fixed_cols = {}
    for k in keys:
        v0 = records[0][k]
        if isinstance(v0, (str, bytes)):
            vals = {r[k] for r in records}
            if len(vals) != 1:
                raise ValueError(f"string column {k!r} differs between records: {sorted(vals)[:5]}")
            constants[k] = v0 if isinstance(v0, str) else v0.decode()
            continue
        g = None if k in per_record else _group_of(k, groups)
        if g is None:
            fixed_cols[k] = np.stack([np.asarray(r[k]) for r in records])
        else:
            var_cols.setdefault(g, {})[k] = [np.asarray(r[k]).reshape(-1) if np.ndim(r[k]) <= 1 else np.asarray(r[k])
                                             for r in records]
    group_meta = {}
    for g, cols in var_cols.items():
        lens = None
        for k, arrs in cols.items():
            l_ = np.array([len(a) for a in arrs], np.int64)
            if lens is None:
                lens = l_
            elif not np.array_equal(lens, l_):
                bad = int(np.flatnonzero(lens != l_)[0])
                raise ValueError(f"column {k!r} of group {g!r} has another length than the group in record {bad}")
            np.save(tmp / f"{k}.npy", np.concatenate(arrs) if len(arrs) else np.zeros(0), allow_pickle=False)
        np.save(tmp / f"{g}_off.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), allow_pickle=False)
        group_meta[g] = dict(offsets=f"{g}_off", columns=sorted(cols))
    for k, v in fixed_cols.items():
        np.save(tmp / f"{k}.npy", v, allow_pickle=False)
    info = dict(format=FORMAT, n=n, groups=group_meta, per_record=sorted(fixed_cols), constants=constants,
                written=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname(), pid=os.getpid())
    info.update(meta or {})
    (tmp / "meta.json").write_text(json.dumps(info, indent=1, default=str))


class Shard:
    """One shard directory, memory-mapped on first use. Every mapped column holds an open file, so a reader of many
    shards must release() the ones it is not using (ShardSet keeps a bounded set open)."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{self.path}: not a {FORMAT} shard")
        self.n = int(self.meta["n"])
        self._a = None

    @property
    def a(self):
        if self._a is None:
            self._a = {p.stem: np.load(p, mmap_mode="r") for p in self.path.glob("*.npy")}
        return self._a

    def release(self):
        self._a = None

    def record(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        a = self.a
        out = dict(self.meta["constants"])
        for g, gm in self.meta["groups"].items():
            off = a[gm["offsets"]]
            s, e = int(off[i]), int(off[i + 1])
            for k in gm["columns"]:
                out[k] = np.array(a[k][s:e])
        for k in self.meta["per_record"]:
            v = np.array(a[k][i])
            out[k] = v.item() if v.ndim == 0 else v
        return out

    def __len__(self):
        return self.n


class ShardSet:
    """All shards under a folder (any depth), indexed as one sequence. Drop-in for the parts of
    ra_pickles.RandomAccessPicklesReader that create_dataset.py used (get_total, get_element, get_multi_in_parallel)."""

    MAX_OPEN = 16          # shards mapped at once (each holds ~40 open files; the usual limit is 1024 per process)

    def __init__(self, folder, record_fn=None):
        self.folder = Path(folder)
        self._open = OrderedDict()
        paths = sorted(p.parent for p in self.folder.rglob("meta.json") if not p.parent.name.endswith(".tmp"))
        self.shards = [Shard(p) for p in paths]
        self.shards = [s for s in self.shards if s.n > 0]
        if not self.shards:
            raise FileNotFoundError(f"no shards under {self.folder}")
        self.cum = np.concatenate([[0], np.cumsum([s.n for s in self.shards])]).astype(np.int64)
        self.record_fn = record_fn

    def get_total(self):
        return int(self.cum[-1])

    __len__ = get_total

    def get_element(self, i):
        i = int(i)
        j = int(np.searchsorted(self.cum, i, side="right") - 1)
        if j in self._open:
            self._open.move_to_end(j)
        else:
            self._open[j] = True
            while len(self._open) > self.MAX_OPEN:
                self.shards[self._open.popitem(last=False)[0]].release()
        r = self.shards[j].record(i - int(self.cum[j]))
        return self.record_fn(r) if self.record_fn else r

    def get_multi_in_parallel(self, indices, timeout=None):
        # memory-mapped reads are fast enough sequentially; the signature matches ra_pickles
        return [self.get_element(i) for i in indices]


# ---- simulations ------------------------------------------------------------------------------------------------
GUN_PREFIX = "gun_"


def sim_record(result, gun=None):
    """A minicalo.simulate_* result (+ the generated gun particle for single-particle simulations) as a flat record."""
    r = {k: v for k, v in result.items()}
    if gun is not None:
        for k, v in gun.items():
            r[GUN_PREFIX + k] = np.asarray(v)
    return r


def sim_from_record(r):
    """Inverse of sim_record: (simulation, gun particle) when the record has a gun, else the simulation alone -- the
    same items ra_pickles held for run_simulations.py's output."""
    gun = {k[len(GUN_PREFIX):]: v for k, v in r.items() if k.startswith(GUN_PREFIX)}
    sim = {k: v for k, v in r.items() if not k.startswith(GUN_PREFIX)}
    for k in ("position", "direction"):            # lists, as particle_generator.ParticleGenerator makes them
        if k in gun:
            gun[k] = np.asarray(gun[k]).tolist()
    return (sim, gun) if gun else sim


def write_sim_shard(path, records, meta=None):
    gun_cols = [k for k in records[0] if k.startswith(GUN_PREFIX)] if records else []
    return write_shard(path, records, SIM_GROUPS, meta=meta, per_record=gun_cols)


def sim_library(folder):
    """Reader of a folder of simulation shards: get_element(i) -> (simulation, gun) or simulation."""
    return ShardSet(folder, record_fn=sim_from_record)


# ---- events -------------------------------------------------------------------------------------------------------
def write_event_shard(path, events, meta=None):
    """events: list of dicts of per-hit arrays (same keys); hit_off.npy gives each event's hit range."""
    return write_shard(path, events, EVENT_GROUPS, meta=meta)


def event_shard(path):
    return Shard(path)
