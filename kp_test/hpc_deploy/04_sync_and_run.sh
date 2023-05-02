#!/bin/bash
rsync -avz -P /home/jg/al_olympics/code/ s5968580@taurus.hrsk.tu-dresden.de:/beegfs/ws/1/s5968580-al_olympics/code --exclude '.git/' --exclude '.mypy_cache/' --exclude '*.pyc' --exclude '__pycache__'
rsync -avz -P /home/mg/exp_results/ s5968580@taurus.hrsk.tu-dresden.de:/beegfs/ws/1/s5968580-al_olympics/exp_results
ssh s5968580@taurus.hrsk.tu-dresden.de << EOF
    cd /beegfs/ws/1/s5968580-al_olympics/code
    export LC_ALL=en_US.utf-8
    export LANG=en_US.utf-8
    sbatch /beegfs/ws/1/s5968580-al_olympics/exp_results/kp_test/02_slurm.slurm
EOF
#    module load Python/3.8.6;