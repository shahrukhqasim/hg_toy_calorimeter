"""Columnar shard storage (replaces ra_pickles and the DeepJetCore .djctd files).

The layout follows the fcc-ai-design shards (lyra/data/pf/<shard>/<array>.npy): a shard is one directory holding one
.npy file per column, plus offset arrays that cut the concatenated columns into records (simulations or events), and a
meta.json. Readers read only a record's rows from each column and close the files again (np.load(..., mmap_mode="r")
works too for interactive use); the files are plain numpy: no pickles, no container needed to read them.

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
            # minicalo returns an empty per-particle list as float64: take the dtype from the non-empty records so one
            # empty record does not turn an integer column (indices, ids) into floats
            full = [x for x in arrs if x.size]
            dt = np.result_type(*full) if full else (arrs[0].dtype if arrs else np.float64)
            np.save(tmp / f"{k}.npy", np.concatenate([x.astype(dt, copy=False) for x in arrs]) if arrs else np.zeros(0, dt),
                    allow_pickle=False)
        np.save(tmp / f"{g}_off.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), allow_pickle=False)
        group_meta[g] = dict(offsets=f"{g}_off", columns=sorted(cols))
    for k, v in fixed_cols.items():
        np.save(tmp / f"{k}.npy", v, allow_pickle=False)
    info = dict(format=FORMAT, n=n, groups=group_meta, per_record=sorted(fixed_cols), constants=constants,
                written=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname(), pid=os.getpid())
    info.update(meta or {})
    (tmp / "meta.json").write_text(json.dumps(info, indent=1, default=str))


class _Column:
    """One .npy column: the header is parsed once, then each read opens the file, reads the rows asked for and closes
    it again, so no file stays open however many shards a reader touches."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read(f)
            self.offset = f.tell()
        if fortran:
            raise ValueError(f"{path}: Fortran-ordered arrays are not supported")
        self.shape, self.dtype = shape, dtype
        self.row = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1

    def rows(self, s, e):
        n = max(0, e - s)
        with open(self.path, "rb") as f:
            f.seek(self.offset + s * self.row * self.dtype.itemsize)
            a = np.fromfile(f, dtype=self.dtype, count=n * self.row)
        if len(a) != n * self.row:
            raise OSError(f"{self.path}: short read")
        return a.reshape((n,) + tuple(self.shape[1:]))


class Shard:
    """One shard directory. Columns are read on demand (no memory maps, no files left open)."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{self.path}: not a {FORMAT} shard")
        self.n = int(self.meta["n"])
        self._cols = None
        self._off = {}

    def _col(self, k):
        if self._cols is None:
            self._cols = {}
        if k not in self._cols:
            self._cols[k] = _Column(self.path / f"{k}.npy")
        return self._cols[k]

    def array(self, k):
        """A whole column (e.g. an offset array, or a per-record column)."""
        c = self._col(k)
        return c.rows(0, c.shape[0] if c.shape else 1)

    def record(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        out = dict(self.meta["constants"])
        for g, gm in self.meta["groups"].items():
            if g not in self._off:                      # offsets are small: kept in memory
                self._off[g] = self.array(gm["offsets"])
            off = self._off[g]
            s, e = int(off[i]), int(off[i + 1])
            for k in gm["columns"]:
                out[k] = self._col(k).rows(s, e)
        for k in self.meta["per_record"]:
            v = self._col(k).rows(i, i + 1)[0]
            out[k] = v.item() if np.ndim(v) == 0 else v
        return out

    def __len__(self):
        return self.n


class ShardSet:
    """All shards under a folder (any depth), indexed as one sequence. Drop-in for the parts of
    ra_pickles.RandomAccessPicklesReader that create_dataset.py used (get_total, get_element, get_multi_in_parallel)."""

    def __init__(self, folder, record_fn=None):
        self.folder = Path(folder)
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
        r = self.shards[j].record(i - int(self.cum[j]))
        return self.record_fn(r) if self.record_fn else r

    def get_multi_in_parallel(self, indices, timeout=None):
        # direct reads are fast enough sequentially; the signature matches ra_pickles
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
