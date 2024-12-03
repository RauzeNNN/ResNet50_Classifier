import os
import shutil
import numpy as np
import yaml
import train
import testBinary
import glob
import pandas as pd

CONFIG_PATH = "/home/ocaki13/ResNet50_Classifier/config.yml"
BATCHSAMPLE = "/home/ocaki13/ResNet50_Classifier/batch_sample_train.png"


def seedTrain(cfg, seedList, deleteImages=True, deleteNonBestModels=True):


    resultsPath =  cfg['dataset_config']['save_dir']
    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    resultsDict = {}
    for seed in seedList:
        currentSaveDir = os.path.join(resultsPath, 'seed_'+str(seed))
        cfg['dataset_config']['save_dir'] = currentSaveDir
        cfg['train_config']['seed'] = seed
        best_path = os.path.join(currentSaveDir, 'models/best.pt')
        train.main(cfg)
        currentResults = testBinary.main(cfg, best_path)  
        resultsDict[seed] = currentResults
        if deleteImages:
            shutil.rmtree(os.path.join(currentSaveDir ,"images"))

        if deleteNonBestModels:
            folder_path = os.path.join(currentSaveDir ,"models")
            files_to_delete = glob.glob(os.path.join(folder_path, "*epoch*"))
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)  # Delete the file
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                
            

    # Convert the dictionary of dictionaries into a DataFrame
    results_df = pd.DataFrame.from_dict(resultsDict, orient='index')

    # Save the DataFrame to a CSV file
    results_df.to_csv(os.path.join(resultsPath,"results.csv"), index_label="Seed")

if __name__ == "__main__":
    #absolute paths suggested especially for wsl.
    with open(CONFIG_PATH, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    seedList = [35, 1063, 306]
    seedTrain(cfg, seedList)