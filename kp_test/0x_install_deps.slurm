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






module load Anaconda3
sh $EBROOTANACONDA3/etc/profile.d/conda.sh
conda create --prefix /beegfs/ws/1/s5968580-al_olympics//al_olympics/conda-env python=3.10
conda activate /beegfs/ws/1/s5968580-al_olympics//al_olympics/conda-env
conda install -c anaconda cython -y
conda install -y -c conda-forge cvxpy pipenv liblapacke
python -m pipenv --python=$(conda run which python) --site-packages install --dev
python -m pipenv shell
LIBACT_BUILD_VARIANCE_REDUCTION=0 python -m pip install git+https://github.com/jgonsior/libact.git









#module load Python/3.8.6
#conda create --name al_olympics_conda python=3.10
conda activate /scratch/ws/1/s5968580-al_olympics/conda-env
#conda install -y -c anaconda cython
#conda install -y -c conda-forge cvxpy pipenv liblapacke
python -m pipenv --python=$(conda run which python) --site-packages install --dev
python -m pipenv shell
pip install git+https://github.com/ntucllab/libact.git -> klappt nicht auf taurus!
MPLCONFIGPATH=/beegfs/ws/1/s5968580-al_olympics/cache /home/s5968580/.local/share/virtualenvs/code-Q_6lMpHQ/bin/python

exit 0