from urllib.request import urlopen
from PIL import Image
import timm
from utils import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--dataset_name",
    type=str,
    required=False,
    choices=["FMoW","COOS","iWildCam","CivilComments","GeoDE","AutoArborist","SelfDrivingCar","FishDetection"],
    default="COOS",
    help="CLIP model type",
)
args = parser.parse_args()

model = timm.create_model(
    'vit_small_patch16_224.dino',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

for split in ['val1','val2','val3','val4','val5','train','test1','test2','test3','test4','test5']:
    dataset = get_dataset(dataset_name=args.dataset_name, split=split, transform=transform)
    print(dataset)
    filename=f'all_datasets/{args.dataset_name}/embeddings/{split}_dino_embeddings.pth'
    dl = DataLoader(dataset, batch_size=64, shuffle=False)
    if not os.path.exists(filename):
        embeddings = []
        uids=[]
        for image, _, _, uid in tqdm(dl):
            print(image.shape)
            print(uid.shape)
            output = model.forward_head(image, pre_logits=True)
            embeddings.append(output)
            uids.append(uid)
        torch.cat(embeddings, dim=0)
        embed_dict = {"embeddings":embeddings, "uids":uids}
        np.save(filename,embed_dict)
