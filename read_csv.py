import os
from PIL import Image
import pandas as pd

df=pd.read_csv('all_datasets/AutoArborist/splits/train.csv')
def can_open_image(path):
    try:
        print(path)
        with Image.open('all_datasets/AutoArborist/'+path) as img:
            img.verify()  # Verify that it's an image
        return True
    except (IOError, FileNotFoundError):
        return False
df = df[df['street_level'].apply(can_open_image)]
df.to_csv('all_datasets/AutoArborist/splits/train_processed.csv')