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
#SBATCH --output=job_logs/$SLURM_JOBID.o 
#SBATCH --error=job_logs/$SLURM_JOBID.e
#
# ===== END SLURM OPTIONS =====
cd $SLURM_SUBMIT_DIR
mkdir -p job_logs
# Scraper for Classical Piano Midi Page
[ ! "$1" ] && echo 'Error: please specify output dir' && exit
dir=$1
base=http://www.piano-midi.de
pages=$(curl -s --max-time 5 $base/midi_files.htm \
    | grep '<tr class="midi"><td class="midi"><a href="' \
    | egrep '[^"]+\.htm' -o)
echo Pages: $pages
mkdir -p $dir
for page in $pages; do
    midis=$(curl -s --max-time 5 $base/$page | egrep '[^"]+format0\.mid' -o)
    for midi in $midis; do
        echo "http://www.piano-midi.de/$midi"
    done | tee /dev/stderr | wget -P $dir -i -
done
cd $dir
ls | egrep -v -i '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm

