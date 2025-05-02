#!/bin/bash

datasets=("GeoDE" "AutoArborist" "SelfDrivingCar")
splits=("train" "test1" "test2" "test3" "test4" "val1" "val2" "val3" "val4")

# Loop through a range of 5 numbers (0 to 4)
for dataset in "${datasets[@]}"; do
  # Nested loop through another range of 5 numbers (0 to 4)
  for split in "${splits[@]}"; do
    sbatch run_linear_probe_embeddings.sh "$dataset" "$split"
  done
done

