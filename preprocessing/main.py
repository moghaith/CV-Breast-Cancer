import cv2
from preprocessing import IMAGE, DATASET
import os 




import io
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import Trainer, TrainingArguments
# import evaluate
from datasets import load_dataset

dataset_path = "D:\\Mine\\Disk\\4_2025_Spring\\CIE552_ComputerVision\\Projects\\Dataset\\POC_Dataset"


def processing():
    SET = DATASET(dataset_path)
    mean, std = SET.norm_parameters()
    # print("mean, std:", mean, std)

    cls_folders = os.listdir(dataset_path)
    for cls_folder in cls_folders:
        images = os.listdir(os.path.join(dataset_path, cls_folder))                
        for img in images:
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(dataset_path, cls_folder, img)
                IMG = IMAGE(img_path)
                read_img = IMG.img_loading()
                if read_img is not None:
                    denoised = IMG.noise_removal(read_img, "NLMD")
                    enhanced = IMG.contrast_enhancement(denoised, "CLAHE")
                    normalized = IMG.zscore_norm(enhanced, mean, std)

    print("DONE")


def vits_classification(dataset_path):
    dataset = load_dataset(dataset_path)

    print(dataset)
    
vits_classification(dataset_path)