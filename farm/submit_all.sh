#!/bin/bash
# Submit the whole production: simulation arrays, then event arrays once their simulations succeeded.
set -e
H=/disk/lhcb_data/sqasim/hgcal_toy; cd $H
sb() { sbatch --parsable "$@"; }
TP=$(sb -J hg_train_particles --array=0-49%25 sim.sbatch train_particles)
TM=$(sb -J hg_train_minbias   --array=0-49%25 sim.sbatch train_minbias)
SP=$(sb -J hg_test_particles  --array=0-4     sim.sbatch test_particles)
SM=$(sb -J hg_test_minbias    --array=0-6     sim.sbatch test_minbias)
ET=$(sb -J hg_train_events --array=0-49%25 --dependency=afterok:$TP:$TM ev.sbatch train_events)
EV=$(sb -J hg_test_events  --array=0-4     --dependency=afterok:$SP:$SM ev.sbatch test_events)
echo "sims: train particles $TP, train minbias $TM, test particles $SP, test minbias $SM; events: train $ET, test $EV" | tee -a $H/logs/submitted.txt
