import cv2
from preprocessing import IMAGE, DATASET
import os 

dataset_path = "/Users/mohamedghaith/Documents/Uni/Year 4/Sem 2/CV/Project/CV-Breast-Cancer/BUSI_split/"
trainset_path = "/Users/mohamedghaith/Documents/Uni/Year 4/Sem 2/CV/Project/CV-Breast-Cancer/BUSI_split/train/" 

def processing():
    SET = DATASET(trainset_path)
    mean, std = SET.norm_parameters()

    dataset = os.listdir(dataset_path)
    for set in dataset:
        set_path = os.path.join(dataset_path, set)
        if not os.path.isdir(set_path):
            continue
        cls_folders = os.listdir(set_path)
        for cls_folder in cls_folders:
            folder_path = os.path.join(set_path, cls_folder)
            try:
                images = os.listdir(folder_path)
            except Exception as e:
                print(f"Error listing directory {folder_path}: {e}")
                continue
                
            for img in images:
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    img_path = os.path.join(folder_path, img)
                    
                    IMG = IMAGE(img_path)
                    read_img = IMG.img_loading()
                    
                    if read_img is not None:
                        denoised = IMG.noise_removal(read_img, "NLMD")  # "NLMD" OR "SRAD"
                        enhanced = IMG.contrast_enhancement(denoised, "CLAHE")  # "HE" OR "CLAHE"
                        normalized = IMG.zscore_norm(enhanced, mean, std)
                    else:
                        print(f"[WARNING] The img is NOT valid: {img_path}")
                        if not os.path.exists(img_path):
                            print(f"File does not exist: {img_path}")
                        else:
                            file_size = os.path.getsize(img_path)
                            print(f"File exists but may be corrupt. Size: {file_size} bytes")

    print("DONE")
processing()
