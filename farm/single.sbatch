#!/bin/bash
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /disk/lhcb_data/sqasim/hgcal_toy/logs/%x_%A_%a.out
# 50 events (1 test particle + 200 PU, full detector) into file k.
set -eo pipefail
H=/disk/lhcb_data/sqasim/hgcal_toy
cd $H/code/python/bin
$H/run.sh python3 create_dataset.py configuration/single_pu200.ini single_pu200 --num-events 50 --first-file ${SLURM_ARRAY_TASK_ID} \
  | sed -E "/^[0-9]+ [0-9]+$|^Remaining|^Loaded in|Building and shuffling|^$/d"
echo "$(date +%T) done"
