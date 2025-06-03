from PIL import Image, ImageFile
import os
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Class image paths
    path_van = os.path.join(base_dir, "data", "van") # 1
    path_truck = os.path.join(base_dir, "data", "truck") # 2
    path_tractor = os.path.join(base_dir, "data", "tractor") # 3
    path_bus = os.path.join(base_dir, "data", "bus") # 4
    path_auto = os.path.join(base_dir, "data", "auto") # 5

    paths = [path_van, path_truck, path_tractor, path_bus, path_auto]
    target_size = (64, 64)

    # Open Dirs
    dirs = []
    for path in paths:
        dir = os.listdir(path)
        dir.sort()
        dirs.append(dir)
    
    data=[]
    labels = []

    for i in range(len(dirs)):
        path = paths[i]
        dir = dirs[i]

        for item in dir:
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):
                im = Image.open(full_path).convert("RGB")
                im = im.resize(target_size)
                im = np.array(im) / 255.0
                data.append(im)
                labels.append(i)
    
    labels = np.array(labels)

    return data, labels

if __name__ == "__main__":
    
    data, labels = load_dataset()
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "vehicle_detection_dataset.npz")
    np.savez(save_path, images=data, labels=labels)