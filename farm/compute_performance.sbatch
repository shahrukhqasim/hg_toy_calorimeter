#!/bin/bash
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /disk/lhcb_data/sqasim/hgcal_toy/logs/%x_%A_%a.out
# 100 events of scenario $1 (pu040 ... pu200) into file k = array index (0-19).
set -eo pipefail
H=/disk/lhcb_data/sqasim/hgcal_toy
echo "$(date +%T) $(hostname -s) $1 task ${SLURM_ARRAY_TASK_ID}"
cd $H/code/python/bin
$H/run.sh python3 create_dataset.py configuration/compute_performance.ini "$1" --num-events 100 --first-file ${SLURM_ARRAY_TASK_ID} \
  | sed -E "/^[0-9]+ [0-9]+$|^Remaining|^Loaded in|Building and shuffling|^$/d"
echo "$(date +%T) done"
