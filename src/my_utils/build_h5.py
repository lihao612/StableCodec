import numpy as np
import os
import h5py
from PIL import Image
from tqdm import tqdm
import glob

with h5py.File('data/train/dataset1.hdf5', 'w') as f:
    idx = 0
    for file in tqdm(os.listdir('data/train/flickr2k/Flickr2K/')):
        img = Image.open(os.path.join('data/train/flickr2k/Flickr2K/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('data/train/div2k/DIV2K_train_HR/')):
        img = Image.open(os.path.join('data/train/div2k/DIV2K_train_HR/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('data/train/clic2020train/')):
        img = Image.open(os.path.join('data/train/clic2020train/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('data/train/clic2020val/')):
        img = Image.open(os.path.join('data/train/clic2020val/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('data/train/clic2021test/')):
        img = Image.open(os.path.join('data/train/clic2021test/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1
    for file in tqdm(os.listdir('data/train/lsdir/')):
        img = Image.open(os.path.join('data/train/lsdir/', file)).convert("RGB")
        f.create_dataset(str(idx), data=img, dtype=np.uint8)
        idx += 1

    