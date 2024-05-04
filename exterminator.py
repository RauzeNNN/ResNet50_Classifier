import numpy as np
from PIL import Image
import os
import sys  

#usage: python3 exterminator.py [directory_path]

def make_top_black(img_path):
    img = np.array(Image.open(img_path))
    if img.shape[0] == 1200 and img.shape[1] == 1600:
        img[:85,:] = 0
        img[:,:240] = 0
        img[-85:,:] = 0
        
    else:
        x = int(img.shape[0]*(40/800))
        img[:x,:] = 0

    last = Image.fromarray(img)
    last.save(img_path)

def iterate_over_all_directory_files(directory_path, function):
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            function(os.path.join(root, filename))

def main():
    n = len(sys.argv)
    if n != 2:
        print("invalid amount of arguments. just need a folder name.")
        return -1
        
    directory_path = sys.argv[1]
    iterate_over_all_directory_files(directory_path, make_top_black)
    return 0


if __name__ == '__main__':
    main()
