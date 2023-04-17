#!/bin/bash
#SBATCH --time=1:59:59   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2210M   # memory per CPU core
#SBATCH -A "p_ml_il"
#SBATCH --output /beegfs/ws/1/s5968580-al_olympics/exp_results/kp_test/02_run_experiment.py_out.txt
#SBATCH --error /beegfs/ws/1/s5968580-al_olympics/exp_results/kp_test/02_run_experiment.py_error.txt
#SBATCH --array 0-108

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

#module load Python/3.8.6

export PYTHONDONTWRITEBYTECODE=1

i=$(( 0 + $SLURM_ARRAY_TASK_ID * 100 ))
#module load Python/3.8.6
end=$(($i+100))
for ((j = $i ; j < $end ; j++)); do
    MPLCONFIGPATH=/beegfs/ws/1/s5968580-al_olympics/cache timeout 30600 /home/s5968580/.local/share/virtualenvs/code-Q_6lMpHQ/bin/python /beegfs/ws/1/s5968580-al_olympics//code//02_run_experiment.py --EXP_TITLE kp_test   --RUNNING_ENVIRONMENT hpc --LOG_FILE log.txt --WORKER_INDEX $j
done
exit 0