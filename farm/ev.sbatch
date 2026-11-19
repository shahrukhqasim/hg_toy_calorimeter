#!/bin/bash
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /disk/lhcb_data/sqasim/hgcal_toy/logs/%x_%A_%a.out
# One task: 200 events of section $1 into files 2k and 2k+1 (k = array index), 8 event-making processes.
set -eo pipefail
H=/disk/lhcb_data/sqasim/hgcal_toy
K=${SLURM_ARRAY_TASK_ID}
echo "$(date +%T) $(hostname -s) section $1 task $K code $(git -C $H/code rev-parse --short HEAD)"
cd $H/code/python/bin
$H/run.sh python3 create_dataset.py configuration/training_g4_11.ini "$1" --num-events 200 --first-file $((2 * K)) \
  | sed -E "/^[0-9]+ [0-9]+$|^Remaining|^Loaded in|Building and shuffling|^$/d"   # sed: exit 0 even if nothing is left
echo "$(date +%T) done"
