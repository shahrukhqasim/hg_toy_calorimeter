#!/bin/bash
#SBATCH -p standard
#SBATCH -c 16
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -o /disk/lhcb_data/sqasim/hgcal_toy/logs/%x_%A_%a.out
# One task: 16 Geant4 workers of run_simulations.py for the section given as $1 (training_g4_11.ini).
# All tasks of a section write their shards into the same folder (names carry the worker seed).
set -eo pipefail
H=/disk/lhcb_data/sqasim/hgcal_toy
echo "$(date +%T) $(hostname -s) section $1 task ${SLURM_ARRAY_TASK_ID} code $(git -C $H/code rev-parse --short HEAD)"
cd $H/code/python/bin
$H/run.sh python3 run_simulations.py configuration/training_g4_11.ini "$1"
echo "$(date +%T) done"
