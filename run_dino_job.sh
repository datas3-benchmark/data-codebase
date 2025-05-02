#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --output=slurm/clip_embeddings-%J.out
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH --time=12:00:00

source /data/vision/beery/scratch/neha/.bashrc
micromamba activate datacomp


datasets=("iWildCam" "GeoDE" "AutoArborist" "SelfDrivingCar")

for dataset_name in "${datasets[@]}"
do
    python baselines/get_dino_embeddings.py --dataset_name $dataset_name
done