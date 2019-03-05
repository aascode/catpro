#!/usr/bin/env bash
#
#SBATCH --time=1:30:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -c4

cd /om2/user/dlow/containers/
module add openmind/singularity/3.0
singularity shell -B /om2/ catpro.simg
cd /home/dlow/catpro/catpro/
python3 gpt2.py
echo 'Finished.'

