import os
import shutil
import numpy as np
import yaml
import train
import testBinary

#absolute paths suggested especially for wsl.
CONFIG_PATH = "/kuacc/users/mharmanli21/eye_classification/ResNet50_Classifier/config.yml"
OUTPUT_PATH = "/kuacc/users/mharmanli21/eye_classification/seed_outputs"
RESULT_PATH = "/kuacc/users/mharmanli21/eye_classification/results_temp"
BATCHSAMPLE = "/kuacc/users/mharmanli21/batch_sample_train.png"
SEED_LIST = [35, 1063, 306, 629, 1940, 288, 399, 1215, 187, 1636]
DELETE_IMAGES = True
DELETE_NON_BEST_MODELS = True

def sort_filenames(l):
    temp = l[:]
    last_flag = False
    if "last_epoch.pt" in temp:
        temp.remove("last_epoch.pt")
        last_flag = True

    #extraction
    for i in range(len(temp)):
        temp[i] = int(temp[i][5:-3])

    temp.sort()

    #packing
    for i in range(len(temp)):
        temp[i] = "epoch" + str(temp[i]) + ".pt"

    if last_flag:
        temp.append("last_epoch.pt")
    return temp

def train_one_seed(cfg, seed):
    global RESULT_PATH
    cfg['train_config']['seed'] = seed

    train.main(cfg)
    best_path = os.path.join(RESULT_PATH + "/models", sort_filenames(os.listdir(RESULT_PATH + "/models"))[-2])
    testBinary.main(cfg, best_path)


def save_results(cfg, seed):
    global OUTPUT_PATH
    global RESULT_PATH
    global SEED_LIST
    global DELETE_IMAGES
    global DELETE_NON_BEST_MODELS

    cfg['train_config']['seed'] = seed

    if DELETE_IMAGES:
        shutil.rmtree(RESULT_PATH + "/images")
    
    if DELETE_NON_BEST_MODELS:
        os.remove(RESULT_PATH + "/models/last_epoch.pt")
        dirpaths = os.listdir(RESULT_PATH + "/models")
        to_be_removed = sort_filenames(dirpaths)[:-1]
        for i in to_be_removed:
            os.remove(RESULT_PATH + "/models/" + i)
    
    seeddir = OUTPUT_PATH + "/seed" + str(seed)
    os.mkdir(seeddir)
    shutil.copytree(RESULT_PATH, seeddir + "/results")
    shutil.copy2(BATCHSAMPLE, seeddir)

    os.remove(BATCHSAMPLE)
    shutil.rmtree(RESULT_PATH)

    f = open(os.path.join(seeddir, "used_config.yml"), "w")
    yaml.dump(cfg, f)
    f.close()

    

if __name__ == "__main__":
    # config_path = 'config.yml'
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    for seed in SEED_LIST:
        train_one_seed(cfg, seed)
        save_results(cfg, seed)
    
    
