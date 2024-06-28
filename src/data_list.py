import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path

def pil_loader(path):
    # Open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    return pil_loader(path)

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path
import argparse  # Import argparse to handle command-line arguments

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

# def make_dataset(dataset_name, image_list, labels):
#     base_path = f'/data2/huwentao/zzy/HashNet-master/pytorch/data/{dataset_name}'
    
#     if labels is not None:
#         if not isinstance(labels, np.ndarray):
#             labels = np.array(labels)  # Convert to numpy array if not already
#         len_ = len(image_list)
#         images = [(os.path.join(base_path, image_list[i].strip()), labels[i]) for i in range(len_)]
#     else:
#         if len(image_list[0].split()) > 2:
#             images = [(os.path.join(base_path, val.split()[0]), np.array([int(la) for la in val.split()[1:]])) for val in image_list]
#         else:
#             images = [(os.path.join(base_path, val.split()[0]), int(val.split()[1])) for val in image_list]
    
#     return images

def make_dataset(dataset_name, image_list, labels=None):
    base_path = f'/data2/huwentao/zzy/HashNet-master/pytorch/data/{dataset_name}'
    images = []

    for val in image_list:
        try:
            parts = val.strip().split()
            if len(parts) < 2:
                print(f"Skipping line due to insufficient parts: {val}")
                continue
            image_path = os.path.join(base_path, parts[0])
            if labels is not None:
                label = labels.pop(0)  # Assuming labels are provided separately and in order
            else:
                label = np.array([int(la) for la in parts[1:]])
            images.append((image_path, label))
        except Exception as e:
            print(f"Error processing line: {val}. Error: {str(e)}")
            continue

    return images



class ImageList(data.Dataset):
    def __init__(self, dataset_name, image_list, labels=None, transform=None, target_transform=None, loader=default_loader):
        self.dataset_name = dataset_name
        self.imgs = make_dataset(dataset_name, image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError(f"Found 0 images for dataset: {dataset_name}")

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        print(f"Loading image from path: {path}")  # 输出路径以验证
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)