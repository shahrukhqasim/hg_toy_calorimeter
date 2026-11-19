# UZH farm production scripts

The scripts that produced the dataset in `/disk/lhcb_data/sqasim/hgcal_toy` (Slurm, `standard` partition). They expect
this repository checked out and built in `$H/code` (`H=/disk/lhcb_data/sqasim/hgcal_toy`) and a venv in `$H/venv`;
copy them to `$H`, or change `H`, to reproduce elsewhere.

- `run.sh`: runs a command in the Geant4 11.4.3 container with the venv and `PYTHONPATH` of the pinned checkout.
- `sim.sbatch SECTION`: one array task of `run_simulations.py` (16 workers) for a section of
  `configuration/training_g4_11.ini`.
- `ev.sbatch SECTION`: one array task of `create_dataset.py`, 200 events into files 2k and 2k+1.
- `submit_all.sh`: the whole production: simulation arrays, then the event arrays once the simulations succeeded
  (10,000 training and 1,000 test events: 60 particles + 200 pileup collisions in a random 30° φ slice).
- `single.sbatch`: test events with one particle + 200 pileup over the full detector
  (`configuration/single_pu200.ini`, 50 events per task): `sbatch -J single_pu200 --array=0-7 single.sbatch`.
- `compute_performance.sbatch SCENARIO`: the pileup scan (one particle + 40 … 200 pileup, full detector, 2,000 events
  each; `configuration/compute_performance.ini`):
  `for s in pu040 pu080 pu120 pu160 pu200; do sbatch -J cp_$s --array=0-19 compute_performance.sbatch $s; done`
