"""Generate a library of Geant4 simulations (single particles, minimum bias or qqbar -> ttbar) on many CPU cores.

  python3 run_simulations.py <config.ini> <section>
  python3 run_simulations.py minbias <output_path> [--cores 10] [--num-simulations N] [--num-simulations-per-file M]

Output: <output_path>/<simtype>_<seed>_<chunk>/ shard directories (calo/shards.py: one .npy per column, offsets per
group, meta.json), each holding num-simulations-per-file simulations, plus <output_path>/detector_spec.json and
<output_path>/sensor_data.npz (minicalo.get_sensor_data(): position, size, layer of every sensor). A shard
is renamed into place only when complete, so an interrupted run leaves no partial shards (only *.tmp directories).
Single-particle shards also hold the generated gun particle (gun_* columns).

num-simulations is per core (as before): the run writes cores x num-simulations simulations.
"""
import argparse
import configparser
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import time

import numpy as np

from calo.config import ensure_tls, pythia_data
from calo.particle_generator import ParticleGenerator
from calo.calo_generator import CaloV3Generator, CaloV4Generator
from calo.shards import sim_record, write_sim_shard

ALLOWED = {'minbias', 'qqbar2ttbar', 'singlepart', 'singlepart_face'}


def _versions():
    out = {}
    for k, cmd in (("geant4", ["geant4-config", "--version"]), ("pythia8", ["pythia8-config", "--version"])):
        try:
            out[k] = subprocess.check_output(cmd, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            out[k] = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out["git_commit"] = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True,
                                                    stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        out["git_commit"] = None
    return out


def _write_once(path, write, suffix=''):
    """Write a file that every job of a production writes with the same content: skip it if it exists, write to a
    per-process temporary file and rename. Many jobs share one output folder, and on a network file system (GlusterFS)
    concurrent renames onto the same name can fail ("Stale file handle"); that only means another job wrote it."""
    if os.path.exists(path):
        return
    tmp = '%s.%s.%d.tmp%s' % (path, socket.gethostname(), os.getpid(), suffix)
    try:
        write(tmp)
        os.replace(tmp, path)
    except OSError as e:
        if not os.path.exists(path):
            raise
        print("note: %s was written by another job (%s)" % (path, e), flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def detector_specs_for(conf):
    def f(k):
        return float(conf[k]) if k in conf else None
    calo_type = conf.get('calo', 'v3')
    CGenerator = {'v3': CaloV3Generator, 'v4': CaloV4Generator}.get(calo_type)
    if CGenerator is None:
        raise NotImplementedError('calo must be v3 or v4, not %s' % calo_type)
    gen = CGenerator(calo_min_kinetic_energy=f('calo_min_kinetic_energy'),
                     world_min_kinetic_energy=f('world_min_kinetic_energy'),
                     calo_max_time=f('calo_max_time'), world_max_time=f('world_max_time'),
                     sensor_thickness=f('sensor_thickness'))
    return calo_type, gen.generate()


def work(worker, simtype, output_path, num_events, num_events_per_file, conf, versions):
    import minicalo

    np.random.seed()
    seeds = [int(x) for x in np.random.randint(0, 1000000000, size=4)]
    print("Worker", worker, "seeds", seeds, flush=True)

    calo_type, detector_specs = detector_specs_for(conf)
    minicalo.initialize(json.dumps(detector_specs), pythia_data(conf), False, *seeds)  # third: collect_full_data
    sensor_file = os.path.join(output_path, 'sensor_data.npz')
    if worker == 0:                                       # the sensor table (calibrate.py starts from it)
        _write_once(sensor_file, lambda p: np.savez(p, **{k: np.asarray(v) for k, v in minicalo.get_sensor_data().items()}),
                    suffix='.npz')

    if 'pdgid' in conf:
        particle_pdgid = [int(x) for x in str(conf['pdgid']).split(',')]
    else:
        particle_pdgid = [11, 22, 211, 111]
        if simtype == 'singlepart':
            particle_pdgid += [15]
    range_energy = [float(conf['energy_min']), float(conf['energy_max'])] if 'energy_min' in conf else [0.1, 200]
    range_eta = [float(conf['eta_min']), float(conf['eta_max'])] if 'eta_min' in conf else [3.1, 1.4]
    particle_generator = ParticleGenerator(detector_specs, range_energy=range_energy, particle_pdgid=particle_pdgid,
                                           range_eta=range_eta)
    filter_deposited = str(conf.get('filter_deposited', 'False')).lower() in ('true', '1', 'yes', 'on')

    num_events_written = 0
    chunk = 0
    while num_events_written < num_events:
        n_file = min(num_events_per_file, num_events - num_events_written)
        records = []
        t0 = time.time()
        while len(records) < n_file:
            if simtype in {'singlepart', 'singlepart_face'}:
                particle = particle_generator.generate(from_iteraction_point=simtype == 'singlepart')
                result = minicalo.simulate_particle(
                    *[float(v) for v in particle['position']], *[float(v) for v in particle['direction']],
                    int(particle['pdgid']), float(particle['energy']))
                result['simulator'] = calo_type
                if filter_deposited:
                    rechit_sum = np.sum(result['rechit_energy'])
                    if rechit_sum < 0.001:
                        continue
                records.append(sim_record(result, particle))
            elif simtype == 'minbias':
                records.append(sim_record(minicalo.simulate_pu()))
            elif simtype == 'qqbar2ttbar':
                records.append(sim_record(minicalo.simulate_qqbar2ttbar()))
        name = "%s_%09d_%05d" % (simtype, seeds[0], chunk)
        meta = dict(kind="simulations", simtype=simtype, calo=calo_type, seeds=seeds, worker=worker, chunk=chunk,
                    config=dict(conf), seconds=round(time.time() - t0, 1), **versions)
        p = write_sim_shard(os.path.join(output_path, name), records, meta=meta)
        num_events_written += len(records)
        chunk += 1
        print("Wrote", p, len(records), "simulations in", round(time.time() - t0, 1), "s", flush=True)


def main(simtype, output_path, cores=10, num_simulations=1000000, num_simulations_per_file=100, conf=None):
    """
    Generate a large set of simulations using multiple CPU cores.

    :param simtype: 1. minbias -- 14 TeV proton-proton interactions
                    2. qqbar2ttbar -- qqbar to ttbar interaction
                    3. singlepart -- single particles from the interaction point with 1.4 < |eta| < 3.1
                    4. singlepart_face -- single particles with 1.4 < |eta| < 3.1, 1 cm away from the calorimeter
    :param output_path: folder for the simulation shards, created if it does not exist
    :param cores: number of worker processes
    :param num_simulations: simulations per worker
    :param num_simulations_per_file: simulations per shard
    """
    conf = dict(conf or {})
    if simtype not in ALLOWED:
        raise ValueError('Wrong simtype %s: select from %s' % (simtype, sorted(ALLOWED)))
    os.makedirs(output_path, exist_ok=True)
    _, spec = detector_specs_for(conf)
    _write_once(os.path.join(output_path, 'detector_spec.json'), lambda p: json.dump(spec, open(p, 'w')))
    versions = _versions()
    print("Going to do", simtype, 'on', cores, 'cores;', versions, flush=True)

    ctx = multiprocessing.get_context('fork')   # Python 3.14 defaults to forkserver on Linux
    processes = []
    for m in range(cores):
        p = ctx.Process(target=work, args=(m, simtype, output_path, num_simulations, num_simulations_per_file, conf,
                                           versions))
        p.start()
        time.sleep(0.3)
        processes.append(p)
    for p in processes:
        p.join()
    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        raise SystemExit("workers %s failed (exit codes %s)" % (failed, [processes[i].exitcode for i in failed]))


if __name__ == '__main__':
    ensure_tls()
    if any(x.endswith('.ini') for x in sys.argv[1:2]):
        config = configparser.ConfigParser()
        config.read(sys.argv[1])
        sec = config[sys.argv[2]]
        main(sec['simtype'], sec['outpath'], int(sec['cores']), int(sec['num-simulations']),
             int(sec['num-simulations-per-file']), sec)
    else:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument('simtype', choices=sorted(ALLOWED))
        ap.add_argument('output_path')
        ap.add_argument('--cores', type=int, default=10)
        ap.add_argument('--num-simulations', type=int, default=1000000)
        ap.add_argument('--num-simulations-per-file', type=int, default=100)
        ap.add_argument('--calo', default='v3', choices=['v3', 'v4'])
        a = ap.parse_args()
        main(a.simtype, a.output_path, a.cores, a.num_simulations, a.num_simulations_per_file, {'calo': a.calo})
