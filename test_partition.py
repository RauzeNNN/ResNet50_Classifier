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
# from sklearn.metrics import auc

image_ext = ['.png', '.jpg']


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
            if '_control' in filename:
                continue
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return natural_sort(image_names)


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

class Results():
    def __init__(self, labels_map, threshold=0.5):
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.labels_map = labels_map
        self.prob_negative = []
        self.prob_positive = []
    def compare(self, y, y_pred_prob):
        prob_postive = y_pred_prob.squeeze()[1].item()
        if y==1:
            self.prob_positive.append(prob_postive)
        else:
            self.prob_negative.append(prob_postive)

        if prob_postive > self.threshold:
            y_pred=1
        else:
            y_pred=0

        if y==1 and y_pred ==1:
            self.tp += 1
        elif y == 0 and y_pred == 1:
            self.fp += 1
        elif y == 1 and y_pred == 0:
            self.fn += 1
        elif y == 0 and y_pred == 0:
            self.tn += 1
    def save_and_print(self, path):
        recall = self.tp / (self.tp + self.fn)
        sensivity = self.tp / (self.tp + self.fn)
        specificity = self.tn / (self.tn + self.fp)
        precision = self.tp / (self.tp + self.fp)
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

        print('tp: {}'.format(self.tp))
        print('fp: {}'.format(self.fp))
        print('tn: {}'.format(self.tn))
        print('fn: {}'.format(self.fn))
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('f1: {}'.format(f1))
        print('specifity: {}'.format(specificity))
        print('sensivity: {}'.format(sensivity))
        print('accuracy: {}'.format(accuracy))

        sns.distplot(self.prob_negative, hist=False, rug=True,
                     color='green', label=self.labels_map[0])
        sns.distplot(self.prob_positive, hist=False, rug=True,
                     color='red', label=self.labels_map[1])
        plt.xlabel('diagnosis confidence')
        plt.legend()
        plt.savefig(os.path.join(path,
                              'prob_dist.png'.format(self.threshold)))
        

def get_detections_from_partitions(model, img, device, partition_size, ROI_flag=False, ROI_SIZE=0.8):
    predictions = []
    h, w = img.shape[:2]

    if ROI_flag:
        roi = int(min(img.shape[:2]) * ROI_SIZE)
        y_max_roi = int(h - 0.1 * w)
        y_min_roi = int(y_max_roi - roi)
        x_min_roi = int(w / 2) - int(roi / 2)
        x_max_roi = int(w / 2) + int(roi / 2)
        # Control panel
        # cv2.rectangle(img, roi_rect[0], roi_rect[1], (255,0,0) ,2)
        # im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(im_rgb)
        # plt.show()

        # roi_rect = [(x_min_roi, y_min_roi), (x_max_roi, y_max_roi)]
        cropped_img = img[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    else:
        y_max_roi = h
        y_min_roi = 0
        x_min_roi = 0
        x_max_roi = w
        cropped_img = img[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    img_h, img_w = cropped_img.shape[:2]
    num_of_partition_w = int(img_w / partition_size) + 1
    num_of_partition_h = int(img_h / partition_size) + 1
    partition_shift_w = (partition_size * num_of_partition_w -
                         img_w) / (num_of_partition_w - 1)
    partition_shift_h = (partition_size * num_of_partition_h -
                         img_h) / (num_of_partition_h - 1)


    partition_rectangles = []
    for c in range(num_of_partition_w):
        for r in range(num_of_partition_h):
            bxmin = int(c * (partition_size - partition_shift_w))
            bxmax = bxmin + partition_size
            bymin = int(r * (partition_size - partition_shift_h))
            bymax = bymin + partition_size
            current_input_img = cropped_img[bymin:bymax, bxmin:bxmax]
            partition_rectangles.append([(bxmin + x_min_roi, bymin + y_min_roi),
                                        (bxmax + x_min_roi, bymax + y_min_roi)])
            current_input_img = pre_process(current_input_img)
            logits = model(current_input_img.to(device))
            softmaxed_scores = model.softmax(logits)

            _, predict = torch.max(softmaxed_scores, 1)
            predictions.append(predict.item())

    out_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    for i, rect in enumerate(partition_rectangles):
        if predictions[i] == 1:
            cv2.rectangle(out_img, rect[0], rect[1], (0, 0, 255), 2)
        else:
            cv2.rectangle(out_img, rect[0], rect[1], (0, 255, 0), 2)
    return out_img


def get_detections_from_partitions_2(model, img, device, partition_size, ROI_flag=False, ROI_SIZE=0.8):
    predictions = []
    h, w = img.shape[:2]
    offset = 80
    cropped_img = img[offset:-offset, offset:-offset]
    partition1 = cropped_img[0:640,0:640]
    partition2 = cropped_img[0:640,640:1280]
    partition3 = cropped_img[640:1280,0:640]
    partition4 = cropped_img[640:1280,640:1280]

    for current_input_img in [partition1, partition2, partition3, partition4]:
        current_input_img = pre_process(current_input_img)
        logits = model(current_input_img.to(device))
        softmaxed_scores = model.softmax(logits)

        _, predict = torch.max(softmaxed_scores, 1)
        predictions.append(predict.item())

    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    out_img = cv2.rectangle(out_img, (offset, offset),
                            (w-offset, h-offset), (0, 0, 0), 1)

    for i, pred in enumerate(predictions):
        if pred == 1 and i == 0:
            xmin, ymin = offset+0, offset+0
            xmax = xmin + 640
            ymax = ymin + 640
            cv2.rectangle(out_img, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), 2)
        if pred == 1 and i == 1:
            xmin, ymin = offset+640, offset+0
            xmax = xmin + 640
            ymax = ymin + 640
            cv2.rectangle(out_img, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), 2)
        if pred == 1 and i == 2:
            xmin, ymin = offset+0, offset+640
            xmax = xmin + 640
            ymax = ymin + 640
            cv2.rectangle(out_img, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), 2)
        if pred == 1 and i == 3:
            xmin, ymin = offset+640, offset+640
            xmax = xmin + 640
            ymax = ymin + 640
            cv2.rectangle(out_img, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), 2)

    return out_img

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

    class_list = os.listdir(test_path)
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


    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]

        img = cv2.imread(img_path, 0)

        out_img = get_detections_from_partitions_2(model, img, device, 640, ROI_flag=False, ROI_SIZE=0.8)
        cv2.imwrite(os.path.join(save_dir, image_name), out_img)
            
         

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    model_path = args.model_path
    # config_path = 'config.yml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    main(cfg, model_path)
