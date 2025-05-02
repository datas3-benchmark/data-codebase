import itertools
import json
import os
import subprocess
import shlex

# Define your lists of parameters
image_dataset_list = ['FMoW','iWildCam']
baselines_list = ["no_filter"]
                #"basic_filter"]
fraction_baselines_list = ["clip_score",
                "random_filter"] 
image_baselines_list = ["image_based"]#,"image_based_intersect_clip_score"]
tasks = ["test1", "test2", "test3", "test4"]
fraction_list = [0.05, 0.10, 0.25, 0.50, 0.75, 0.95]


# for dataset in image_dataset_list:
#     embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
#     centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
#     for baseline in ["image_based"]:
#         for task in tasks:
#             val_embedding_path = f"all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"
#             save_folder = f"experiments/{dataset}/{baseline}/"
#             save_path= save_folder+f"{task}_subset.npy"
#             if not os.path.exists(save_path):
#                 subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s"'%(baseline, embedding_path, save_path, 1, val_embedding_path, centroids_path)))
#             if not os.path.exists(save_folder+f"test{task[4]}_lora_metrics.json"):
#                 subprocess.call(shlex.split('sbatch baselines/run_lora_finetune.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

# image_clip
for dataset in image_dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
    for baseline in ['image_alignment']:#'image_clip'
        for task in tasks:
            val_embedding_path = f"all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"
            for fraction in fraction_list:
                save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
                save_path= save_folder+f"{task}_subset.npy"
                if not os.path.exists(save_path):
                    subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s"'%(baseline, embedding_path, save_path, fraction, val_embedding_path, centroids_path)))
                if not os.path.exists(save_folder+f"test{task[4]}_lora_metrics.json"):
                    subprocess.call(shlex.split('sbatch baselines/run_lora_finetune.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))
        
# baselines_list    
for dataset in image_dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    for baseline in baselines_list:
         save_folder = f"experiments/{dataset}/{baseline}/"
         save_path= save_folder+"subset.npy"
         if not os.path.exists(save_path):
             subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s'%(baseline, embedding_path, save_path, 1)))
         if not os.path.exists(save_folder+"test1_lora_metrics.json"):
             subprocess.call(shlex.split('sbatch baselines/run_lora_finetune.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

#fraction_baselines list
for dataset in image_dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    for baseline in ['random_filter']:
        for fraction in fraction_list:
             save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
             save_path= save_folder+"subset.npy"
             if not os.path.exists(save_path):
                 subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s'%(baseline, embedding_path, save_path, fraction)))
             if not os.path.exists(save_folder+"test1_lora_metrics.json"):
                 subprocess.call(shlex.split('sbatch baselines/run_lora_finetune.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

#match_dist
for dataset in image_dataset_list:
    baseline = "match_dist"
    for fraction in fraction_list:
        for task in tasks:
            task_num=int(task[4])
            save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
            save_path= save_folder+f"{task}_subset.npy"
            if not os.path.exists(save_path):
                subprocess.call(shlex.split('sbatch run_csv_baseline.sh "%s" "%s" "%s" %s "%s"'%(baseline, dataset, task_num, fraction, save_path)))
            if not os.path.exists(save_folder+f"test{task[4]}_lora_metrics.json"):
                subprocess.call(shlex.split('sbatch baselines/run_lora_finetune.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

# # match_label
# for dataset in ['iWildCam','FMoW']:
#     baseline = "match_label"
#     for task in tasks:
#         task_num=int(task[4])
#         save_folder = f"experiments/{dataset}/{baseline}/"
#         save_path= save_folder+f"{task}_subset.npy"
#         if not os.path.exists(save_path):
#             subprocess.call(shlex.split('sbatch run_csv_baseline.sh "%s" "%s" "%s" %s "%s"'%(baseline, dataset, task_num, 1, save_path)))
#         if not os.path.exists(save_folder+f"test{task[4]}_lora_metrics.json"):
#             subprocess.call(shlex.split('sbatch baselines/run_lora_finetune.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

    
        
    



