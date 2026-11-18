"""Create an event dataset from simulation libraries (run_simulations.py output): every event samples particle
simulations (and pileup simulations) without replacement, adds noise, applies the calibration and the noise cut, and
assigns truth.

  python3 create_dataset.py <config.ini> <section>

Reads the simulation shards under particles_folder / pu_folder and writes event shards to output_folder
(<output_folder>/<NNNNN>/, calo/dataset_creator.py documents the columns; read them with calo.shards.event_shard).
The calibration file (calibration_data, or $HG_TOY_CALORIMETER_CALIBRATION_DATA) is the .npz that calibrate.py writes;
the old gzip-pickled sensor_data_*_calibrated.bin files are still read.
"""
import configparser
import gzip
import json
import os
import pickle
import sys

import numpy as np
from calo.dataset_creator import DatasetCreator
from calo.shards import sim_library


def str2bool(v):
    return v.lower() in ("yes", "true", "on", "1")


def load_sensor_data(path):
    if path.endswith('.npz'):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    with gzip.open(path, 'rb') as f:            # legacy format (trusted file: pickle)
        return pickle.load(f)


def library_or_none(folder, needed):
    if not needed and (not folder or folder.lower() == 'none' or not os.path.isdir(folder)):
        return None
    try:
        return sim_library(folder)
    except FileNotFoundError:
        if needed:
            raise
        return None


if __name__ == '__main__':
    config_file = sys.argv[1]
    section = sys.argv[2]
    config = configparser.ConfigParser()
    config.read(config_file)
    sec = config[section]

    particles_folder = sec['particles_folder']
    pu_folder = sec.get('pu_folder', 'none')
    output_folder = sec['output_folder']
    num_events_total = int(sec['num_events_total'])

    cut = np.float64(sec['noise_cut']) if 'noise_cut' in sec else 0.0009
    print("Using cut: ", cut)
    include_tracks = str2bool(sec['include_tracks']) if 'include_tracks' in sec else True
    if not include_tracks:
        print("NOT INCLUDING TRACKS IN THIS SIMULATION!")

    num_cores = int(sec['num_event_creation_cores']) if 'num_event_creation_cores' in sec else 4
    num_parallel_reading_threads = int(sec.get('num_parallel_reading_threads', 20))  # unused: reads are memory-mapped

    if 'calibration_data' in sec:
        calib_path = sec['calibration_data']
    elif 'HG_TOY_CALORIMETER_CALIBRATION_DATA' in os.environ:
        calib_path = os.getenv('HG_TOY_CALORIMETER_CALIBRATION_DATA')
    else:
        raise RuntimeError('Calibration data not found: set calibration_data in the config or '
                           'HG_TOY_CALORIMETER_CALIBRATION_DATA (calibrate.py writes the file)')
    sensor_data = load_sensor_data(calib_path)

    read_int = lambda s: int(sec[s])
    read_float = lambda s: float(sec[s])
    if 'num_particles' in sec:
        num_particles = int(sec['num_particles'])
    else:
        num_particles = lambda: min(read_int('num_particles_max'), max(read_int('num_particles_min'), int(
            np.random.normal(read_int('num_particles_mean'), read_int('num_particles_std')))))

    noise_sigma = float(sec['noise_sigma']) if 'noise_sigma' in sec else 1.2e-5
    noise_mean = float(sec['noise_mean']) if 'noise_mean' in sec else 0.0
    alter_muon_truth = str(sec['alter_muon_truth']) if 'alter_muon_truth' in sec else False

    sample_isolated_particles = None
    if 'sample_isolated_particles_dist' in sec:
        sample_isolated_particles_dist = float(sec['sample_isolated_particles_dist'])
        if 'sample_isolated_particles_N' in sec:
            sample_isolated_particles_N = lambda: int(sec['sample_isolated_particles_N'])
        elif all(k in sec for k in ['sample_isolated_particles_N_mean', 'sample_isolated_particles_N_std',
                                    'sample_isolated_particles_N_min', 'sample_isolated_particles_N_max']):
            sample_isolated_particles_N = lambda: min(read_int('sample_isolated_particles_N_max'), max(
                read_int('sample_isolated_particles_N_min'), int(np.random.normal(
                    read_int('sample_isolated_particles_N_mean'), read_int('sample_isolated_particles_N_std')))))
        elif 'sample_isolated_particles_N_mean' in sec:
            raise ValueError("Missing one or more of the following keys in configuration file: "
                             "'sample_isolated_particles_N_mean', 'sample_isolated_particles_N_std', "
                             "'sample_isolated_particles_N_min', 'sample_isolated_particles_N_max'")
        else:
            raise ValueError("'sample_isolated_particles_N' must be found in configuration file if "
                             "'sample_isolated_particles_dist' is present")
        sample_isolated_particles = (sample_isolated_particles_N, sample_isolated_particles_dist)

    noise_fluctuations = ('type_a', noise_mean, noise_sigma)
    num_pu = int(sec['num_pu'])

    for i in range(5):
        print(num_particles if type(num_particles) is int else num_particles(), num_pu)

    os.makedirs(output_folder, exist_ok=True)
    num_events_per_file = (read_int('num_events_per_djc_file') if 'num_events_per_djc_file' in sec else
                           read_int('num_events_per_file') if 'num_events_per_file' in sec else
                           min(200, num_events_total))
    pu_phase_cut = read_float('pu_phase_cut') if 'pu_phase_cut' in sec else None
    compute_spectators_dist = str2bool(sec['compute_spectators_dist']) if 'compute_spectators_dist' in sec else True

    particles_iterator = library_or_none(particles_folder, needed=True)
    pu_iterator = library_or_none(pu_folder, needed=num_pu > 0)
    print("Simulations: %d particles, %d pileup" % (particles_iterator.get_total(),
                                                    pu_iterator.get_total() if pu_iterator else 0))

    meta = dict(config_file=os.path.abspath(config_file), section=section, settings=dict(sec),
                calibration_data=os.path.abspath(calib_path))
    with open(os.path.join(output_folder, 'dataset_config.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    dataset_creator = DatasetCreator(pu_iterator=pu_iterator,
                                     particles_iterator=particles_iterator,
                                     output_path=output_folder,
                                     rechit_cut=cut,
                                     sensor_data=sensor_data,
                                     num_events_per_djc_file=num_events_per_file,
                                     num_particles_per_event=num_particles,
                                     num_pu_per_event=num_pu,
                                     num_events_total=num_events_total,
                                     pu_phase_cut=pu_phase_cut,
                                     min_hits_cut=1, compute_spectators_dist=compute_spectators_dist,
                                     num_event_creation_processes=num_cores,
                                     noise_fluctuations=noise_fluctuations,
                                     num_parallel_reading_threads=num_parallel_reading_threads,
                                     include_tracks=include_tracks,
                                     sample_isolated_particles=sample_isolated_particles,
                                     alter_muon_truth=alter_muon_truth,
                                     meta=meta)
    dataset_creator.process()
