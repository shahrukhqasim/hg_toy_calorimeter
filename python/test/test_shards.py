"""Round trip of the shard storage (numpy only, no Geant4):  cd python && PYTHONPATH=libs python3 -m unittest discover -s test"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calo.shards import event_shard, sim_library, sim_record, write_event_shard, write_sim_shard


def fake_sim(rng, n_rec, n_hit, n_part, gun=True):
    r = {"simtype": "singlepart", "simulator": "v4",
         "rechit_energy": rng.random(n_rec), "rechit_idx": rng.integers(0, 1000, n_rec),
         "hit_particle_id": rng.integers(0, n_part, n_hit), "hit_particle_deposit": rng.random(n_hit),
         "hit_particle_sensor_idx": rng.integers(0, 1000, n_hit),
         "particles_pdgid": rng.integers(-300, 300, n_part), "particles_kinetic_energy": rng.random(n_part)}
    g = dict(position=[0.0, 0.0, 0.0], direction=list(rng.random(3)), energy=float(rng.random()), pdgid=22, id="gamma")
    return r, (g if gun else None)


class ShardTest(unittest.TestCase):
    def test_sim_round_trip(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            sims = [fake_sim(rng, n, 2 * n, k) for n, k in ((5, 2), (0, 1), (17, 4))]
            write_sim_shard(Path(d) / "a", [sim_record(s, g) for s, g in sims[:2]], meta=dict(seeds=[1, 2, 3, 4]))
            write_sim_shard(Path(d) / "sub" / "b", [sim_record(s, g) for s, g in sims[2:]])
            lib = sim_library(d)
            self.assertEqual(lib.get_total(), 3)
            for i, (s, g) in enumerate(sims):
                s2, g2 = lib.get_element(i)
                for k, v in s.items():
                    self.assertTrue(np.array_equal(np.asarray(v), np.asarray(s2[k])), k)
                self.assertEqual(g2["pdgid"], 22)
                self.assertEqual(g2["id"], "gamma")
                self.assertEqual(g2["direction"], g["direction"])
                self.assertAlmostEqual(g2["energy"], g["energy"])
            self.assertEqual(json.loads((Path(d) / "a" / "meta.json").read_text())["seeds"], [1, 2, 3, 4])
            self.assertFalse(any(p.name.endswith(".tmp") for p in Path(d).rglob("*")))

    def test_minbias_without_gun(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            s, _ = fake_sim(rng, 3, 4, 2, gun=False)
            s["simtype"] = "pu"
            write_sim_shard(Path(d) / "pu", [sim_record(s)])
            self.assertIsInstance(sim_library(d).get_element(0), dict)

    def test_events(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as d:
            evs = [{"recHitFeatures": rng.random((n, 10)).astype(np.float32), "t_idx": rng.integers(-1, 5, n).astype(np.int32),
                    "t_pos": rng.random((n, 3)).astype(np.float32)} for n in (7, 1, 30)]
            sh = event_shard(write_event_shard(Path(d) / "00000", evs))
            self.assertEqual(sh.n, 3)
            self.assertTrue(np.array_equal(np.load(Path(d) / "00000" / "hit_off.npy"), [0, 7, 8, 38]))
            for i, e in enumerate(evs):
                r = sh.record(i)
                for k in e:
                    self.assertTrue(np.array_equal(r[k], e[k]), k)
                    self.assertEqual(r[k].dtype, e[k].dtype)

    def test_many_shards_no_open_files(self):
        """Two libraries of 60 shards read in random order under a 64-file limit (pileup mixing reads particles and
        pileup together: memory-mapped columns ran into the open-files limit)."""
        import resource
        rng = np.random.default_rng(3)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        with tempfile.TemporaryDirectory() as d:
            libs, sims = [], []
            for name in ("parts", "pu"):
                ss = [fake_sim(rng, 4, 6, 2) for _ in range(120)]
                for k in range(60):
                    write_sim_shard(Path(d) / name / f"s{k:03d}", [sim_record(s, g) for s, g in ss[2 * k:2 * k + 2]])
                libs.append(sim_library(Path(d) / name))
                sims.append(ss)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, hard), hard))
            try:
                for i in rng.permutation(120):
                    for lib, ss in zip(libs, sims):
                        s2, _ = lib.get_element(i)
                        self.assertTrue(np.array_equal(s2["rechit_energy"], ss[i][0]["rechit_energy"]))
                        self.assertTrue(np.array_equal(s2["particles_pdgid"], ss[i][0]["particles_pdgid"]))
            finally:
                resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    def test_empty_record_keeps_integer_dtype(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as d:
            a, _ = fake_sim(rng, 3, 4, 2, gun=False)
            b = dict(a, particles_pdgid=np.array([]), particles_kinetic_energy=np.array([]))   # what minicalo returns
            a["particles_pdgid"] = a["particles_pdgid"].astype(np.int32)
            write_sim_shard(Path(d) / "s", [sim_record(b), sim_record(a)])
            r = sim_library(d).get_element(1)
            self.assertEqual(r["particles_pdgid"].dtype, np.int32)

    def test_rejects_ragged_group(self):
        with tempfile.TemporaryDirectory() as d:
            bad = {"simtype": "x", "rechit_energy": np.zeros(3), "rechit_idx": np.zeros(2)}
            with self.assertRaises(ValueError):
                write_sim_shard(Path(d) / "bad", [bad])
            self.assertEqual(list(Path(d).iterdir()), [])     # neither the shard nor its .tmp directory remains


if __name__ == "__main__":
    unittest.main()
