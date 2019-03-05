#!/usr/bin/env bash
#
#SBATCH --time=00:55:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -c4

module add openmind/singularity/3.0
singularity exec -B /om2/ -e /om2/user/dlow/containers/catpro.simg /home/dlow/catpro/catpro/gpt2_no_ellie.py
echo 'Finished.'

