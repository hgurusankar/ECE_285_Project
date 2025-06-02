from PIL import Image
import os, sys
import cv2
import numpy as np

def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_vehicle = os.path.join(base_dir, "data", "vehicles")
    path_notvehicle = os.path.join(base_dir, "data", "non-vehicles")
    dirs_vehicle = os.listdir(path_vehicle)
    dirs_notvehicle = os.listdir(path_notvehicle)
    dirs_vehicle.sort()
    dirs_notvehicle.sort()
    data=[]
    labels = []

    # Label = 1 if vehicle
    for item in dirs_vehicle:
        if os.path.isfile(path_vehicle+item):
            im = Image.open(path_vehicle+item).convert("RGB")
            im = np.array(im)
            data.append(im)
            labels.append(1)

    for item in dirs_notvehicle:
        if os.path.isfile(path_notvehicle+item):
            im = Image.open(path_notvehicle+item).convert("RGB")
            im = np.array(im)
            data.append(im)
            labels.append(0)

    labels = np.array(labels)

    return data, labels

if __name__ == "__main__":
    
    data, labels = load_dataset()
    data = np.array(data)
    labels = np.array(labels)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "vehicle_detection_dataset.npz")
    np.savez(save_path, images=data, labels=labels)