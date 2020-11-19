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
# Scraper for Yamaha e-Piano Competition dataset
[ ! "$1" ] && echo 'Error: please specify output dir' && exit
dir=$1
pages='http://www.piano-e-competition.com/ecompetition/midi_2002.asp
http://www.piano-e-competition.com/ecompetition/midi_2004.asp
http://www.piano-e-competition.com/ecompetition/midi_2006.asp
http://www.piano-e-competition.com/ecompetition/midi_2008.asp
http://www.piano-e-competition.com/ecompetition/midi_2009.asp
http://www.piano-e-competition.com/ecompetition/midi_20011.asp
'
mkdir -p $dir
for page in $pages; do
    for midi in $(curl -s $page | egrep -i '[^"]+\.mid' -o | sed 's/^\/*/\//g'); do
        echo "http://www.piano-e-competition.com$midi"
    done
done | wget -P $dir -i -
cd $dir
ls | egrep -v -i '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm

