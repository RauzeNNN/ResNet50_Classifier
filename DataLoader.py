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
import glob
import torchio as tio

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif', '.PNG', '.tiff']

def RadiologyAugmentationTIO(image, transforms_dict):    
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image,(0,-1))),  # Add channel and batch dim
    )  
    # Apply augmentations
    transform = tio.OneOf(transforms_dict)
    transformed_subject = transform(subject)
    
    transformed_image = transformed_subject["image"].data.numpy()[0,:,:,0]
    return transformed_image

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
    def __init__(self, data_path, labels_map, ch=1, input_size=(512, 512), augmentation=False):
        super(Data_Classifier, self).__init__()
        self.channel = ch
        self.augmentation = augmentation
        self.output_size = input_size
        self.class_list = []
        for lbl in labels_map:
            self.class_list.append(labels_map[lbl]) 
        print('Class list:')
        print(self.class_list)
        self.image_list, self.label_list = self.get_data(data_path)
        self.Counter = 0
        # Define augmentation pipeline IMGAUG.
        self.transforms_dict = {
            tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=40): 0.1,
            tio.transforms.RandomElasticDeformation(num_control_points=5, max_displacement=20, locked_borders=1): 0.1,
            tio.transforms.RandomAnisotropy(axes=(0, 1), downsampling=(2, 4)): 0.1,
            tio.transforms.RandomBlur(): 0.1,
            tio.transforms.RandomGhosting(): 0.1,
            tio.transforms.RandomSpike(num_spikes = 1, intensity= (1, 2)): 0.1,
            tio.transforms.RandomBiasField(coefficients = 0.2, order= 3): 0.1,
            tio.RandomGamma(log_gamma=0.1): 0.1,
        }
        
    def transform_mask(self, image):
        if self.augmentation == True:
            if random.random() > 0.5:
                image = RadiologyAugmentationTIO(image, self.transforms_dict)
                self.Counter += 1
                cv2.imwrite(os.path.join('deneme/','torchio'+str(self.Counter)+'.png'),image)

        if len(image.shape)==2:
            h, w = image.shape
            if h != self.output_size[0] or w != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)  # why not 3?
        else:
            h, w, c = image.shape
            if h != self.output_size[0] or w != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w,1), order=3)  # why not 3?
            
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
        
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext and '_label' not in filename:
                    label = maindir.split('/')[-1]
                    label_idx = self.class_list.index(label)
                    label_arr = (np.arange(len(self.class_list)) ==
                                    label_idx).astype(np.float32)
                    image_paths.append(apath)
                    labels.append(label_arr)
        return image_paths, np.array(labels)


