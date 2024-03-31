import os
import shutil
import numpy as np
import yaml
import train
import testBinary

#absolute paths suggested especially for wsl.
global CONFIG_PATH = "/home/rauzen/Projects/eye_project/classifier/config.yml"
global OUTPUT_PATH = "/mnt/c/Users/mbomt/Downloads/deneme"
global RESULT_PATH = "/home/rauzen/Projects/eye_project/classifier/results"
global SEED_LIST = [35, 1063]
global DELETE_IMAGES = True
global DELETE_NON_BEST_MODELS = True

def train_one_seed(cfg, seed)
    cfg['train_config']['seed'] = seed
    train.main(cfg)
    best_path = sorted(os.listdir(RESULT_PATH + "/models"))[-2]
    print("testing: "+ best_path)
    testBinary.main(cfg, best_path)


def save_results(cfg, seed):
    cfg['train_config']['seed'] = seed
    if DELETE_IMAGES:
        os.rmdir(RESULT_PATH + "/images")
    
    if DELETE_NON_BEST_MODELS:
        os.remove(RESULT_PATH + "/models/last_epoch.pt")
        dirpaths = os.listdir(RESULT_PATH + "/models")
        for i in sorted(dirpaths)[:-1]:
            os.remove(i)
    
    seeddir = OUTPUT_PATH + "/seed" + seed
    os.makedir(seeddir)
    shutil.copytree(RESULT_PATH, seeddir + "/results")
    os.copy(RESULT_PATH + "/../batch_sample_train.png", seeddir)

    os.remove(RESULT_PATH + "/../batch_sample_train.png")
    os.rmdir(RESULT_PATH)

    f = file(os.path.join(RESULT_PATH, "used_config.yml"), "w")
    yaml.dump(cfg, f)

    

if __name__ == "__main__":
    # config_path = 'config.yml'
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    for i in SEED_LIST:
        train_one_seed(cfg, seed)
        save_results(cfg, seed)
    
    