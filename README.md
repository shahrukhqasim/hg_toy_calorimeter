## Warning
Not ready yet. Use with caution.

# A High Granularity Toy Calorimeter
The source code for a toy calorimeter to study particle reconstruction performance.

This calorimeter was also employed in the following paper:
https://link.springer.com/article/10.1140/epjc/s10052-022-10665-7


Cite as:
```
@article{Qasim_2022,
	doi = {10.1140/epjc/s10052-022-10665-7},
	url = {https://doi.org/10.1140%2Fepjc%2Fs10052-022-10665-7},
	year = 2022,
	month = {aug},
	publisher = {Springer Science and Business Media {LLC}},
	volume = {82},
	number = {8},
	author = {Shah Rukh Qasim and Nadezda Chernyavskaya and Jan Kieseler and Kenneth Long and Oleksandr Viazlo and Maurizio Pierini and Raheel Nawaz},
	title = {End-to-end multi-particle reconstruction in high occupancy imaging calorimeters with graph neural networks},
	journal = {The European Physical Journal C}
}
```

## Install / Compile
The code builds against a current Geant4 (tested: 11.4.3) and Pythia 8.3 (tested: 8.318), with pybind11 from pip and
jsoncpp from the submodule. ROOT and fastjet are no longer needed. Any environment with `geant4-config`,
`pythia8-config`, CMake >= 3.16, a C++17 compiler and Python 3 works (tested in the `cudabear_blackwell_oct_2026_base`
apptainer image, Python 3.14).

```
git clone https://github.com/shahrukhqasim/hg_toy_calorimeter.git
cd hg_toy_calorimeter
git submodule update --init jsoncpp
pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build build -j 16          # -> build/lib/minicalo.cpython-*.so
```

Porting notes (Geant4 10.7 -> 11, Pythia 8.2/8.30x -> 8.318):
- `g4root.hh` no longer exists: `G4AnalysisManager.hh`.
- Pythia's external random engine is a `shared_ptr`; Pythia's data directory is now the argument of
  `minicalo.initialize` (it was hard-coded); a failed `Pythia::init()` now raises instead of passing silently.
- `ParticleDecays:allowPhotonRadiation` was removed from Pythia 8.3 (it made `init()` fail) and is dropped.
- `Tune:preferLHAPDF = 0`: the tune's PDF from Pythia's own grids (LHAPDF is not installed).
- The calibration constants depend on the Geant4 version: rerun `calibrate.py` (below) instead of reusing
  `sensor_data_v3_calibrated.bin`.

## Generating data
As described in the paper, there are two steps to generate a dataset with this calorimeter.
First individual simulations are generated which can be either single particle simulations
or proton-proton interactions. A dataset of simulations is generated and then to generate
events, simulations are sampled for every event without replacement.

### Storage format
Simulations and events are stored as columnar shards (`python/libs/calo/shards.py`), the layout of the fcc-ai-design
training shards: one directory per shard with one `.npy` per column, `<group>_off.npy` offset arrays that cut the
concatenated columns into records, and a `meta.json` (record count, columns, Geant4/Pythia versions, seeds, config).
They are plain numpy files: read them anywhere with numpy (memory-mapped), no pickles, no container. This replaces
`ra_pickles` (simulations) and the DeepJetCore `.djctd` files (events); TensorFlow and DeepJetCore are no longer used.

```python
from calo.shards import sim_library, event_shard
sims = sim_library('data/mix')            # all simulation shards under a folder
sim, gun = sims.get_element(0)            # minicalo result dict (+ the gun particle for single-particle samples)
ev = event_shard('data/events/00000')     # one event shard
hits = ev.record(0)                       # recHitFeatures [N, 10], t_idx, t_energy, t_pos [N, 3], ...
```

### Setting environment:
```
export PYTHONPATH=$PYTHONPATH:$PWD/build/lib:$PWD/python/libs
```
Geant4 11 needs a larger static-TLS reserve when it is loaded as a Python module; the scripts set
`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=3145728` themselves (`calo.config.ensure_tls`).

### Generate simulations
```
cd python/bin
python3 run_simulations.py configuration/production.ini single_mix     # or: python3 run_simulations.py minbias <out> --cores 1
```
Each worker writes shards `<outpath>/<simtype>_<seed>_<chunk>/` of `num-simulations-per-file` simulations, plus
`<outpath>/detector_spec.json` and `<outpath>/sensor_data.npz` (the sensor table).

### Calibrate
```
python3 run_simulations.py configuration/production.ini sim_calibration_dataset_em_v4
python3 run_simulations.py configuration/production.ini sim_calibration_dataset_had_v4
python3 calibrate.py --sensor-data <em outpath>/sensor_data.npz --em <em outpath> --had <had outpath> --out calib_v4.npz
```

### Generate events
```
python3 create_dataset.py configuration/datasets.ini <section>     # calibration_data = the calib_*.npz file
```
Writes event shards `<output_folder>/<NNNNN>/` (`hit_off.npy` and per-hit `recHitFeatures` [N, 10] with the same
columns as before, `t_idx`, `t_energy`, `t_pos`, `t_time`, `t_pid`, `t_spectator`, `t_fully_contained`, `t_rec_energy`,
`t_is_unique`) and `dataset_config.json`. `num_events_per_djc_file` is still accepted (alias `num_events_per_file`).

`configuration/quick_test.ini` runs the whole chain in a few minutes; `configuration/calibration_g4_11.ini` makes the
calibration samples.

### Tests
```
cd python && PYTHONPATH=libs python3 -m unittest discover -s test     # storage round trip, numpy only
```

## Repository layout
```
CMakeLists.txt, src/, include/   the Geant4 simulation (minicalo Python module)
jsoncpp/                         submodule (detector description parsing)
python/libs/calo/                detector generator, particle gun, event mixing, shard storage
python/bin/                      run_simulations.py, calibrate.py, create_dataset.py, extract_sensor_data.py
python/bin/configuration/        production.ini / datasets.ini (production settings), quick_test.ini,
                                 calibration_g4_11.ini
python/test/                     tests
```
The Geant4 10 example macros, ROOT scripts, old container recipes, the resolution/plotting scripts and the calibration
notebook (all tied to ra_pickles / DeepJetCore / TensorFlow) were removed; they are in the history before this branch.
