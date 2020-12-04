#!/bin/bash
# ======= SLURM OPTIONS ======= (user input required)
#
### Specify queue to run
#SBATCH --partition=GPU
### Set the job name
#SBATCH --job-name=gen_music
### Specify the # of cpus for your job.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --mem=10GB
#SBATCH --time=300:00:00
### pass the full environment
#SBATCH --export=ALL
#SBATCH --output=job_logs/%j.o 
#SBATCH --error=job_logs/%j.e
#
# ===== END SLURM OPTIONS =====
cd $SLURM_SUBMIT_DIR
mkdir -p job_logs
mkdir -p archived_logs
# Remember to activate conda env
nvidia-smi
python -u train.py -c config/base.yml config/train.yml -m archived_logs/$SLURM_JOB_ID