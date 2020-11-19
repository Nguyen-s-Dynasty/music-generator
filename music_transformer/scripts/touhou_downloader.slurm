#!/bin/bash
# ======= SLURM OPTIONS ======= (user input required)
#
### Specify queue to run
#SBATCH --partition=Orion
### Set the job name
#SBATCH --job-name=music_download
### Specify the # of cpus for your job.
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --mem=10GB
#SBATCH --time=30:00:00
### pass the full environment
#SBATCH --export=ALL
#SBATCH --output=job_logs/touhou_download_$SLURM_JOBID.o 
#SBATCH --error=job_logs/touhou_download_$SLURM_JOBID.e
#
# ===== END SLURM OPTIONS =====
cd $SLURM_SUBMIT_DIR
[ ! "$1" ] && echo 'Error: please specify output dir' && exit
dir=$1
for url in $(curl -s https://thwiki.cc/%E5%88%86%E7%B1%BB:%E5%AE%98%E6%96%B9MIDI \
    | egrep -o '[^"]+?\.mid' \
    | egrep '^/' \
    | sed 's/^/https:\/\/thwiki.cc/g' \
    | uniq);
do url=$(curl -s "$url" \
    | egrep -o '[^"]+?\.mid' \
    | egrep '^/' \
    | grep -v '%' \
    | sed 's/^/https:/g' \
    | uniq);
echo $url | tee /dev/stderr
done | uniq | wget -P $dir -i -
cd $dir
ls | egrep -i -v '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm

