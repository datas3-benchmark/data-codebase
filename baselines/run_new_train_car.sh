#!/bin/bash
#SBATCH --partition=csail-shared
#SBATCH --qos=lab-free
#SBATCH --account=vision-beery
#SBATCH --output=slurm/slurm-%J.out
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH --time=12:00:00

source /data/vision/beery/scratch/neha/.bashrc
conda activate datacomp
 

python baselines/train_on_subset_car.py \
	--dataset_name "$1" \
	--subset_path "$2" \
    --outputs_path "$3" \
    --dataset_config "$4" \
    --lr $5 \
    --finetune_type "$6" \
    --batch_size $7 \
    --checkpoint_path "$8"\