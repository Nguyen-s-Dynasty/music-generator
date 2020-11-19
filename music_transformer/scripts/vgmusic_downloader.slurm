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
# Scraper for Video Game music dataset - Thanks to @bbeckwi2 for finding :')
[ ! "$1" ] && echo 'Error: please specify output dir' && exit
dir=$1
pages='https://www.vgmusic.com/music/console/nintendo/nes/
https://www.vgmusic.com/music/console/nintendo/gameboy/
https://www.vgmusic.com/music/console/nintendo/snes/
https://www.vgmusic.com/music/console/nintendo/n64/
https://www.vgmusic.com/music/console/nintendo/virtualboy/
https://www.vgmusic.com/music/console/nintendo/gamecube/
https://www.vgmusic.com/music/console/nintendo/ds/
https://www.vgmusic.com/music/console/nintendo/3ds/
https://www.vgmusic.com/music/console/nintendo/wii/
https://www.vgmusic.com/music/console/nintendo/wiiu/
https://www.vgmusic.com/music/console/nintendo/switch/
https://www.vgmusic.com/music/console/sega/master/
https://www.vgmusic.com/music/console/sega/gamegear/
https://www.vgmusic.com/music/console/sega/segacd/
https://www.vgmusic.com/music/console/sega/32x/
https://www.vgmusic.com/music/console/sega/saturn/
https://www.vgmusic.com/music/console/sega/dreamcast/
https://www.vgmusic.com/music/console/sony/ps1/
https://www.vgmusic.com/music/console/sony/ps2/
https://www.vgmusic.com/music/console/sony/ps3/
https://www.vgmusic.com/music/console/sony/ps4/
https://www.vgmusic.com/music/console/sony/psp/
https://www.vgmusic.com/music/console/microsoft/xbox/
https://www.vgmusic.com/music/console/microsoft/xbox360/
https://www.vgmusic.com/music/console/microsoft/xboxone/
https://www.vgmusic.com/music/console/nec/tg16/
https://www.vgmusic.com/music/console/nec/tduo/
https://www.vgmusic.com/music/console/nec/sgx/
https://www.vgmusic.com/music/console/nec/pcfx/
https://www.vgmusic.com/music/console/snk/neogeo/
https://www.vgmusic.com/music/console/snk/neogeopocket/
https://www.vgmusic.com/music/console/atari/2600/
https://www.vgmusic.com/music/console/atari/7800/
https://www.vgmusic.com/music/console/atari/lynx/
https://www.vgmusic.com/music/console/mattel/intellivision/
https://www.vgmusic.com/music/console/coleco/colecovision/
https://www.vgmusic.com/music/console/magnavox/odyssey2/
https://www.vgmusic.com/music/console/3do/3do/
https://www.vgmusic.com/music/console/philips/cd-i/
https://www.vgmusic.com/music/computer/microsoft/windows/
https://www.vgmusic.com/music/computer/commodore/commodore/
https://www.vgmusic.com/music/computer/miscellaneous/msx/
https://www.vgmusic.com/music/computer/atari/atari/
https://www.vgmusic.com/music/computer/commodore/amiga/
https://www.vgmusic.com/music/computer/amstrad/amstradcpc/
https://www.vgmusic.com/music/computer/apple/appleii/
https://www.vgmusic.com/music/computer/apple/macintosh/
https://www.vgmusic.com/music/computer/nec/pc-88/
https://www.vgmusic.com/music/computer/nec/pc-98/
https://www.vgmusic.com/music/computer/tomy/tutor/
https://www.vgmusic.com/music/computer/sharp/x68000/
https://www.vgmusic.com/music/computer/sinclair/spectrum/
https://www.vgmusic.com/music/other/miscellaneous/arcade/
https://www.vgmusic.com/music/other/miscellaneous/medley/
https://www.vgmusic.com/music/other/miscellaneous/piano/
https://www.vgmusic.com/music/other/miscellaneous/comedy/
'
mkdir -p $dir
for page in $pages; do
    for midi in $(curl -s $page | egrep -i '[^"]+\.mid' -o | sed 's/^\/*/\//g'); do
        echo "$page$midi"
    done
done | wget -P $dir -i -
cd $dir
ls | egrep -v -i '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm

