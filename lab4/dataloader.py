import pandas as pd
from torch.utils import data
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import PIL
import random


def getData(mode, root):
    if mode == 'train':
        img = pd.read_csv(root + 'train_img.csv')
        label = pd.read_csv(root + 'train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(root + 'test_img.csv')
        label = pd.read_csv(root + 'test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, file_root, img_root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        # self.root = root
        self.file_root = file_root
        self.img_root = img_root
        self.img_name, self.label = getData(mode, file_root)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.img_root + self.img_name[index] + '.jpeg'
        label = self.label[index]

        img = Image.open(path).convert('RGB')
        if self.mode == 'train':
            transform_method = transforms.Compose([
                transforms.RandomCrop(480),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif self.mode == 'test':
            transform_method = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            print("invalid mode!")

        return transform_method(img), label
        # return img, label
