import numpy as np
import cv2
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
import yaml
from scipy.ndimage.interpolation import zoom

# from sklearn.metrics import auc

image_ext = ['.bmp', '.png', '.jpg', '.tiff', '.tif', '.PNG']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='the config path')
    ap.add_argument('model_path', help='model path')

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
            if ext in image_ext:
                image_names.append(apath)
    return natural_sort(image_names)


# def pre_process(img):
#     img = np.float32(img)
#     img = (img - img.mean()) / img.std()
#     # HW to CHW (for gray scale)
#     img = np.expand_dims(img, 0)
#     img = np.expand_dims(img, 0)

#     # HWC to CHW, BGR to RGB (for three channel)
#     # img = img.transpose((2, 0, 1))[::-1]
#     img = torch.as_tensor(img)
#     return img

def pre_process(img_org, img_size ):
    if len(img_org.shape)==2:
        imgHeight, imgWidth = img_org.shape
        if imgHeight != img_size[0] or imgWidth != img_size[1]:
            img_input = zoom(img_org, (img_size[0] / imgHeight, img_size[1] / imgWidth), order=3)  
        else:
            img_input = img_org
    else:
        imgHeight, imgWidth, c = img_org.shape
        if imgHeight != img_size[0] or imgWidth != img_size[1]:
            img_input = zoom(img_org, (img_size[0] / imgHeight, img_size[1] / imgWidth, 1), order=3)  
    #z normalizization
    mean3d = np.mean(img_input, axis=(0,1))
    std3d = np.std(img_input, axis=(0,1))
    img_input = (img_input-mean3d)/std3d
    if len(img_input.shape)==2:
        img_input = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().cuda()
    else:
        img_input = img_input.transpose((2, 0, 1))[::-1]
        img_input = torch.from_numpy(img_input.astype(np.float32)).unsqueeze(0).float().cuda()
    
    # # Transform to tensor
    # img_input = TF.to_tensor(img_input)
    # # Normalize
    # img_input = TF.normalize(img_input,[0.5], [0.5]).unsqueeze(0).cuda()

    return img_input

class Results():
    def __init__(self, labels_map, threshold=0.5):
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.labels_map = labels_map
        self.res_dict = {}
        self.prob_negative = []
        self.prob_positive = []
        for cls in labels_map:
            self.res_dict[cls] = {'tp':0, 'fp':0, 'tn':0, 'fn':0}

    def compare(self, y, y_pred_prob):
        _, predict = torch.max(y_pred_prob, 1)
        predict_name = self.labels_map[predict.item()]

        prob_postive = y_pred_prob.squeeze()[1].item()
        if y==1:
            self.prob_positive.append(prob_postive)
        else:
            self.prob_negative.append(prob_postive)

        if prob_postive > self.threshold:
            y_pred=1
        else:
            y_pred=0

        #print("----y, ypred")
        #print(y, y_pred)
        #print("----")
        if y==1 and y_pred ==1:
            self.tp += 1
        elif y == 0 and y_pred == 1:
            self.fp += 1
        elif y == 1 and y_pred == 0:
            self.fn += 1
        elif y == 0 and y_pred == 0:
            self.tn += 1

    def save_and_print(self, path):
        epsilon = 1e-9
        recall = self.tp / (self.tp + self.fn+epsilon)
        sensivity = self.tp / (self.tp + self.fn+epsilon)
        specificity = self.tn / (self.tn + self.fp+epsilon)
        print(self.tp, self.fp)
        precision = self.tp / (self.tp + self.fp+epsilon)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        result_path = os.path.join(path,
                            'results_threshold_{}.txt'.format(self.threshold))
        with open(result_path, 'w') as f:
            f.write('tp: {}\n'.format(self.tp))
            f.write('fp: {}\n'.format(self.fp))
            f.write('tn: {}\n'.format(self.tn))
            f.write('fn: {}\n'.format(self.fn))
            f.write('precision: {}\n'.format(precision))
            f.write('recall: {}\n'.format(recall))
            f.write('f1: {}\n'.format(f1))
            f.write('specifity: {}\n'.format(specificity))
            f.write('sensivity: {}\n'.format(sensivity))
            f.write('accuracy: {}\n'.format(accuracy))

        sns.displot(self.prob_negative, hist=False, rug=True,
                    color='green', label=self.labels_map[0], diag_kind="kde")
        sns.displot(self.prob_positive, hist=False, rug=True,
                    color='red', label=self.labels_map[1], diag_kind="kde")
        plt.xlabel('diagnosis confidence')
        plt.legend()
        plt.savefig(os.path.join(path,
                            'prob_dist.png'.format(self.threshold)))

def main(cfg, model_path):
    # model configs
    input_size = (cfg['model_config']['input_size'][0],
                  cfg['model_config']['input_size'][1])
    num_class = cfg['model_config']['num_class']
    ch = cfg['model_config']['channel']

    # train configs
    use_cuda = cfg['train_config']['use_cuda']

    # dataset configs
    test_path = cfg['dataset_config']['test_path']
    image_list = get_image_list(test_path)
    output_save_dir = cfg['dataset_config']['save_dir']
    save_dir = os.path.join(output_save_dir,'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    class_list = natural_sort(os.listdir(test_path))
    labels_map = {}
    for i, cls in enumerate(class_list):
        labels_map[i] = cls

    model = ResNet50Classifier(ch, num_class, use_cuda)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if use_cuda:
        print('Gpu available')
        print(torch.cuda.get_device_name(0))
        device = "cuda:0"
        dtype = torch.cuda.FloatTensor
        model.to(device=device)
    else:
        model.to(device="cpu")

    correct = 0
    results = Results(labels_map)
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]

        img_org = cv2.imread(img_path,0)

        img = pre_process(img_org, input_size)
        logits = model(img.to(device))
        softmaxed_scores = model.softmax(logits)

        _, predict = torch.max(softmaxed_scores, 1)
        predict_name = labels_map[predict.item()]
        gt_label = img_path.split('/')[-2]
        gt_label_idx = class_list.index(gt_label)
        results.compare(gt_label_idx, softmaxed_scores)
        if gt_label_idx == predict.item():
            correct += 1
            plt.title('{}, {}'.format(gt_label, predict_name), color='green')
            plt.axis("off")
            plt.imshow(img_org, cmap="gray")
            plt.savefig(os.path.join(save_dir,image_name))
            plt.clf()
        else:
            plt.title('{}, {}'.format(gt_label, predict_name), color='red')
            plt.axis("off")
            plt.imshow(img_org, cmap="gray")
            plt.savefig(os.path.join(save_dir,image_name))
            plt.clf()
    try:
        results.save_and_print(output_save_dir)
    except:
        print("save and print did not worked")
        
    print('Accuracy:', correct/len(image_list))
         

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    model_path = '/kuacc/users/ocaki13/hpc_run/workfolder/classifierExp1Sgd/models/epoch14.pt'
    # config_path = 'config.yml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    main(cfg, model_path)
