from PIL import Image, ImageFile
import os
import numpy as np
from scipy.io import loadmat

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    train = os.path.join(base_dir, "data", "cars_train", "cars_train")
    label_train = os.path.join(base_dir, "data", "labels", "cars_train")

    target_size = (112, 112)

    # Open Dirs
    dir = os.listdir(train)
    dir.sort()

    # Load Mat Files
    label_train_dict = loadmat(label_train)

    label_dict = {}
    labels = label_train_dict['annotations'][0]
    for i in range(len(labels)):
        label_dict[labels[i][5][0]] = labels[i][4][0]

    data = []
    labels = []


    i = 0
    for item in dir:
        img_path = os.path.join(train, item)
        if os.path.isfile(img_path):
            im = Image.open(img_path).convert("RGB")
            im = im.resize(target_size)
            im = np.array(im) / 255.0
            data.append(im)
            labels.append(label_dict[item])
            i = i+1
    
    labels = np.array(labels)

    return data, labels

if __name__ == "__main__":
    
    data, labels = load_dataset()
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "stanford_cars_dataset.npz")
    np.savez(save_path, images=data, labels=labels)