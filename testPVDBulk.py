import numpy as np
import os
from tqdm import tqdm
import re
import torch
from Model import ResNet50Classifier
import seaborn as sns
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import time
# from sklearn.metrics import auc
image_ext = ['.bmp', '.png', '.jpg', '.tiff', '.tif', '.PNG']
CH=1
NUM_CLASS=2
USE_CUDA=True

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('test_path', help='path of the folder in which test images are located')
    ap.add_argument('model_path', help='path of the folder in which .pt models are located')
    ap.add_argument('save_dir', help='path of the folder in which results will be saved')

    args = ap.parse_args()
    return args


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext and 'PVD' in filename:
                image_names.append(apath)
    return natural_sort(image_names)

def get_model_list(path):
    model_list = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in ['.pt']:
                model_list.append(apath)
    return natural_sort(model_list)


def pre_process(img):
    img = np.float32(img)
    img = (img - img.mean()) / img.std()
    # HW to CHW (for gray scale)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)

    # HWC to CHW, BGR to RGB (for three channel)
    # img = img.transpose((2, 0, 1))[::-1]
    img = torch.as_tensor(img)
    return img


def testPVDBulk(current_model, image_list, class_list, save_dir):
    #create save_Dir for current seed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    model = ResNet50Classifier(CH, NUM_CLASS, USE_CUDA)
    model.load_state_dict(torch.load(current_model))
    model.eval()
    if USE_CUDA:
        print('Gpu available')
        print(torch.cuda.get_device_name(0))
        device = "cuda:0"
        dtype = torch.cuda.FloatTensor
        model.to(device=device)
    else:
        model.to(device="cpu")

    correct = 0
    times = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]

        img_org = cv2.imread(img_path,0)
        s = time.time()
        img = pre_process(img_org)
        logits = model(img.to(device))
        softmaxed_scores = model.softmax(logits)

        _, predict = torch.max(softmaxed_scores, 1)
        times += time.time() - s

        predict_name = labels_map[predict.item()]
        gt_label = img_path.split('/')[-2]
        gt_label_idx = class_list.index(gt_label)

        
        if gt_label_idx == predict.item():
            correct += 1
            if gt_label_idx == 1:
                tp+=1
            else:
                tn+=1
        else:
            if gt_label_idx == 1:
                fn+=1
            else:
                fp+=1
            plt.title('{}, {}'.format(gt_label, predict_name), color='red')
            plt.axis("off")
            plt.imshow(img_org, cmap="gray")
            plt.savefig(os.path.join(save_dir,image_name))
            plt.clf()
            
    epsilon = 1e-9
    precision = tp/(tp+fp+epsilon)
    recall = tp/(tp+fn+epsilon)
    resultsDict = {
        'Accuracy': round(correct/len(image_list),4)*100,
        'recall:': round(recall,4)*100,
        'precision': round(precision,4)*100,
        'f1':round(2 * precision * recall / (precision + recall+epsilon), 4)*100
    }
    
    # Calculate average time per run in milliseconds
    average_time_ms = (times / len(image_list)) * 1000
    print(f"Average inference time: {average_time_ms:.3f} ms")
    
    return resultsDict

if __name__ == "__main__":
    args = parse_args()
    test_path = args.test_path
    test_images = get_image_list(test_path)
    model_path = args.model_path
    model_list = get_model_list(model_path)
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    #extract labels
    class_list = natural_sort(os.listdir(test_path))
    labels_map = {}
    for i, cls in enumerate(class_list):
        labels_map[i] = cls
        
    resultsDict = {}
    for current_model in model_list:
        current_seed = current_model.split('/')[-4]
        current_results = testPVDBulk(current_model, test_images, class_list, os.path.join(save_dir,current_seed))
        resultsDict[current_seed] = current_results

    # Convert the dictionary of dictionaries into a DataFrame
    results_df = pd.DataFrame.from_dict(resultsDict, orient='index')

    # Save the DataFrame to a CSV file
    results_df.to_csv("results.csv", index_label="Seed")