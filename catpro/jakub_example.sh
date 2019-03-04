#!/usr/bin/env bash
#
#SBATCH --time=7-00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -c4

module add openmind/singularity/3.0

IMAGE='/om2/user/jakubk/meningioma/containers/meningioma_tensorflow-1.12.0-gpu-py3.sif'
singularity exec \
  --nv \
  --bind /om:/om:ro \
  --bind /om2/user/jakubk/nobrainer-training-data:/om2/user/jakubk/nobrainer-training-data:ro \
  --bind /om2/user/jakubk/meningioma \
  $IMAGE \
  python "$@"

echo 'Finished.'