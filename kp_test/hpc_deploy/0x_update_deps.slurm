#!/bin/bash
#SBATCH --time=100:59:59  # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1875M   # memory per CPU core
#SBATCH -A "p_ml_il"
#SBATCH --output /beegfs/ws/1/s5968580-al_olympics/exp_results/slurm_pipenv__out.txt
#SBATCH --error /beegfs/ws/1/s5968580-al_olympics/exp_results/slurm_pipenv__error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE








MPLCONFIGPATH=/beegfs/ws/1/s5968580-al_olympics/cache /home/s5968580/.local/share/virtualenvs/code-Q_6lMpHQ/bin/python -m pipenv update
MPLCONFIGPATH=/beegfs/ws/1/s5968580-al_olympics/cache /home/s5968580/.local/share/virtualenvs/code-Q_6lMpHQ/bin/python -m pip uninstall libact -y
LIBACT_BUILD_VARIANCE_REDUCTION=0 MPLCONFIGPATH=/beegfs/ws/1/s5968580-al_olympics/cache /home/s5968580/.local/share/virtualenvs/code-Q_6lMpHQ/bin/python -m pip install git+https://github.com/jgonsior/libact.git

exit 0