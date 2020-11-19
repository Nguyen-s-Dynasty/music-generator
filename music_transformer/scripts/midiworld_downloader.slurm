#!/bin/bash
# ======= SLURM OPTIONS ======= (user input required)
#
### Specify queue to run
#SBATCH --partition=Orion
### Set the job name
#SBATCH --job-name=music_download
### Specify the # of cpus for your job.
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1,mem=8GB
#SBATCH --time=30:00:00
### pass the full environment
#SBATCH --export=ALL
#SBATCH --output=job_logs/$SLURM_JOBID.o 
#SBATCH --error=job_logs/$SLURM_JOBID.e
#
# ===== END SLURM OPTIONS =====

cd $SLURM_SUBMIT_DIR
mkdir -p job_logs

[ ! "$1" ] && echo 'Error: please specify output dir' && exit
[ ! "$2" ] && echo 'Error: please specify page url' && exit

echo "$(curl -s $2 | egrep -o 'http.+download/[^"]+' | uniq)" \
    | wget --content-disposition -P $1 -i -

cd $1
ls | egrep -i -v '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm
