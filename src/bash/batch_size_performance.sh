#!/bin/bash
#SBATCH --time=2-00:00:00  				                  # Wall time
#SBATCH --array=0-38					                      # Number of startet main.py (One per dataset)
#SBATCH --nodes=1   					                      # number of nodes
#SBATCH --ntasks=1      				                    # limit to one node
#SBATCH --cpus-per-task=2  				                  # number of processor cores (i.e. threads)
#SBATCH --partition=romeo				                    # partition
#SBATCH --mem-per-cpu=20000M   		                  # memory per CPU core
#SBATCH -A p_sp_bigdata 				                    # name of the associated project
#SBATCH -J "Performance" 				                    # name of the job
#SBATCH -o "Performance-%A_%a.out"                  # output file name (std out)
#SBATCH --error="Performance-%A_%a.err" 			      # error file name (std err)
#SBATCH --mail-user="vincent.melisch@tu-dresden.de" # will be used to used to update you about the state of your job
#SBATCH --mail-type ALL

module purge
module load Python/3.10.4

source /home/vime121c/Workspaces/scratch/vime121c-db-project/env/bin/activate

pip install --upgrade pip
pip install pandas
pip install scikit-learn
pip install numpy
pip install matplotlib

python /home/vime121c/Workspaces/scratch/vime121c-db-project/Data-Mining/src/clustering/evaluation/batch_size_performance.py $SLURM_ARRAY_TASK_ID

deactivate
