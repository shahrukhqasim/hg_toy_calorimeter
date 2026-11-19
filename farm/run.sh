#!/bin/bash
# Run a command in the Geant4 11.4.3 container with this production's venv and code (pinned git checkout).
#   /disk/lhcb_data/sqasim/hgcal_toy/run.sh python3 run_simulations.py ...
source /disk/users/sqasim/apptainer_shared/env.sh
H=/disk/lhcb_data/sqasim/hgcal_toy
apptainer exec -B /disk/users -B /disk/lhcb_data -B /scratch /disk/users/sqasim/images/cudabear_blackwell_oct_2026_base.sif \
  bash -c "source $H/venv/bin/activate && export PYTHONPATH=$H/code/build/lib:$H/code/python/libs OPENBLAS_NUM_THREADS=1 && cd \"\$PWD\" && \"\$@\"" bash "$@"
