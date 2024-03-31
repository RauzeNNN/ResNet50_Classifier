import os
import shutil
import numpy as np
import yaml
import train
import testBinary

#absolute paths suggested especially for wsl.
CONFIG_PATH = "/home/rauzen/Projects/eye_project/classifier/config.yml"
OUTPUT_PATH = "/mnt/c/Users/mbomt/Downloads/deneme"
RESULT_PATH = "/home/rauzen/Projects/eye_project/classifier/results"
SEED_LIST = [35, 1063]
DELETE_IMAGES = True
DELETE_NON_BEST_MODELS = True

def train_one_seed(cfg, seed):
    global RESULT_PATH
    cfg['train_config']['seed'] = seed

    train.main(cfg)
    best_path = os.path.join(RESULT_PATH + "/models",sorted(os.listdir(RESULT_PATH + "/models"))[-2])
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
        for i in sorted(dirpaths)[:-1]:
            os.remove(RESULT_PATH + "/models/" + i)
    
    seeddir = OUTPUT_PATH + "/seed" + str(seed)
    os.mkdir(seeddir)
    shutil.copytree(RESULT_PATH, seeddir + "/results")
    shutil.copy2(RESULT_PATH + "/../batch_sample_train.png", seeddir)

    os.remove(RESULT_PATH + "/../batch_sample_train.png")
    shutil.rmtree(RESULT_PATH)

    f = open(os.path.join(seeddir, "used_config.yml"), "w")
    yaml.dump(cfg, f)

    

if __name__ == "__main__":
    # config_path = 'config.yml'
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    for seed in SEED_LIST:
        train_one_seed(cfg, seed)
        save_results(cfg, seed)
    
    