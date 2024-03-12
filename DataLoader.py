import torch
from torch.utils.data import Dataset
import os
import re
from torchvision import transforms
import numpy as np
import cv2
import random
import torchvision.transforms.functional as TF
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

image_ext = ['.bmp', '.png', '.jpg', '.tiff', '.tif', '.PNG']


def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    return image


def random_rotate(image):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    return image

class Data_Classifier(Dataset):
    def __init__(self, data_path, ch=1, input_size=(512, 512), augmentation=False):
        super(Data_Classifier, self).__init__()
        self.image_list, self.label_list = self.get_data(data_path)
        self.channel = ch
        self.augmentation = augmentation
        self.height = input_size[0]
        self.width = input_size[1]

    def transform_mask(self, image):

        if random.random() > 0.5:
            image = random_rot_flip(image)
        elif random.random() > 0.5:
            image = random_rotate(image)

        if len(image.shape)==2:
            y, x = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y), order=3)  # why not 3?
        else:
            y, x, c = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y,1), order=3)  # why not 3?
            
        #z normalizization
        mean3d = np.mean(image, axis=(0,1))
        std3d = np.std(image, axis=(0,1))
        image = (image-mean3d)/std3d
        if len(image.shape)==2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:        
            image = image.transpose((2, 0, 1))[::-1]
            image = torch.from_numpy(image.astype(np.float32))

        return image

    def __getitem__(self, index):

        # read image
        imgPath = self.image_list[index]
        y = self.label_list[index]
        y = torch.from_numpy(y)
        img = cv2.imread(imgPath, 0)

        # Preprocess
        img = self.transform_mask(img)


        return img, y, imgPath

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_data(self, path):
        image_paths = []
        labels = []
        class_list = self.natural_sort(os.listdir(path))
        print(class_list)
        
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext and '_pred' not in filename:
                    label = maindir.split('/')[-1]
                    label_idx = class_list.index(label)
                    label_arr = (np.arange(len(class_list)) ==
                                    label_idx).astype(np.float32)
                    image_paths.append(apath)
                    labels.append(label_arr)
        return image_paths, np.array(labels)


