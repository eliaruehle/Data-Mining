#!/bin/bash
cd /beegfs/ws/1/s5968580-al_olympics//code
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
module load Python/3.8.6

create_synthetic_training_data_id=1

train_imital_id=1
alipy_eva=$(sbatch --parsable --dependency=afterok:$create_synthetic_training_data_id:$train_imital_id /beegfs/ws/1/s5968580-al_olympics//code//05_alipy_eva.slurm)
exit 0