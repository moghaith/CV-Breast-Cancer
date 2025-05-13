import cv2
from preprocessing import IMAGE, DATASET
import os 

dataset_path = "D:\\Mine\\Disk\\4_2025_Spring\\CIE552_ComputerVision\\Projects\\Dataset\\" # write your dir path
trainset_path = "D:\\Mine\\Disk\\4_2025_Spring\\CIE552_ComputerVision\\Projects\\Dataset\\train" # write your dir path
def processing():
    SET = DATASET(trainset_path)
    mean, std = SET.norm_parameters()

    dataset = os.listdir(dataset_path)
    for set in dataset:
        ds = os.listdir(os.path.join(dataset_path, set))
        
        cls_folders = os.listdir(ds)
        for cls_folder in cls_folders:
            images = os.listdir(os.path.join(dataset_path, ds, cls_folder))                
            
            for img in images:
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    img_path = os.path.join(dataset_path, cls_folder, img)
                    IMG = IMAGE(img_path)
                    read_img = IMG.img_loading()
                    if read_img is not None:
                        denoised = IMG.noise_removal(read_img, "NLMD")  # "NLMD" OR "SRAD"
                        enhanced = IMG.contrast_enhancement(denoised, "CLAHE")  # "HE" OR "CLAHE"
                        normalized = IMG.zscore_norm(enhanced, mean, std)

    print("DONE")

