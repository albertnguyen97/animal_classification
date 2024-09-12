import os
import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, ColorJitter
from PIL import Image

import timeit

# Mở và lưu lại ảnh để loại bỏ hồ sơ màu
# image = Image.open('img_srgb.png')
# image.save('img_rgb.png')
start = timeit.default_timer()
class AnimalDataset(Dataset):
    def __init__(self, root="./data/animals_v2", train=True, transform=None):
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        if train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        self.image_paths = []
        self.labels = []

        for category in self.categories:
            category_path = os.path.join(data_path, category)
            print(os.listdir(category_path))
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(image_path)
                    self.labels.append(self.categories.index(category))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    transform = Compose([
        ToTensor(), # chuyen doi ve c x h x w
        Resize((224, 224)) # moi anh co 1 kich thuoc
    ])

    dataset = AnimalDataset(root="./data/animals_v2", train=True, transform=transform)
    image, label = dataset.__getitem__(12347)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=6)
    for images, labels in dataloader:
        print(images.shape)
        print(labels)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
