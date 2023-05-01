#!/bin/bash
#SBATCH --time=1:59:59   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2210M   # memory per CPU core
#SBATCH -A "p_ml_il"
#SBATCH --output /beegfs/ws/1/s5968580-al_olympics//slurm_kp_test__out.txt
#SBATCH --error /beegfs/ws/1/s5968580-al_olympics//slurm_kp_test__error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

#module load Python/3.8.6
zip -r /beegfs/ws/1/s5968580-al_olympics/exp_results/kp_test.zip /beegfs/ws/1/s5968580-al_olympics/exp_results/kp_test
exit 0