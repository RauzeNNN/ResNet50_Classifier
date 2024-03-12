from torchvision.datasets.utils import download_url
import tarfile
import os

data_dir = './data/cifar10'

if os.path.exists(data_dir):
    print("Data directory already exists")
    
else:
    
    os.makedirs(data_dir)
    # Dowload the dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')
    
    # Extract from archive
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

    print(os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    print(classes)
