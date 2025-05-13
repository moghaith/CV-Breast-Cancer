import cv2
import os
import numpy as np
from pathlib import Path
from SRAD import SRAD

class IMAGE:
    
# make these functions in one class
# class preprocessing:
#     __init__():
    def __init__(self, img_path, train_set_path=None):
        self.img_path = img_path
        self.img_title = Path(img_path).stem
        self.train_set_path = train_set_path
        self.mean = None
        self.std = None
        if train_set_path:
            self.mean, self.std = self.norm_parameters()

    
    def img_loading(self):
            # Try to handle special characters in path
        try:
            read_img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            if read_img is not None:
                return read_img
            else:
                print(f"The img is NOT valid:\n{self.img_path}")
                # Try using cv2.samples.findFile for better path handling
                alt_path = cv2.samples.findFile(self.img_path)
                if alt_path:
                    read_img = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)
                    if read_img is not None:
                        return read_img
                return None
        except Exception as e:
            print(f"Error loading image {self.img_path}: {e}")
            return None


    
    def saving_img(self, img, task, output_folder=None, output_format="png"):
        if output_folder is not None:
            os.makedirs(os.path.join(output_folder, "preprocessed_images", os.path.basename(os.path.dirname(self.img_path))), exist_ok=True)
            cv2.imwrite(os.path.join(output_folder, f"{self.img_title}_{task}.{output_format}"), img)
            
        else:
            output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.img_path))), "preprocessed_images", task, os.path.basename(os.path.dirname(self.img_path)))
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(os.path.join(output_folder, f"{self.img_title}_{task}.{output_format}"), img)
    
    
    def noise_removal(self, read_img, method_code, saving=True):
        # NLMD: Non-Local Mean Denoising
        # img_title = read_img.split(".")[:-1]
        if method_code.upper() == "NLMD":
            denoised_img = cv2.fastNlMeansDenoising(read_img, h=10, templateWindowSize=7, searchWindowSize=21)
            
        # SRAD: Speckle Reducing Anisotropic Diffusion
        elif method_code.upper() == "SRAD":
            denoised_img = SRAD(read_img, 200, 0.05, 1)
    
        # if parameter was invalid
        else:
            print("Please enter valid parameter(s)")
            return None
            
        if saving:
            self.saving_img(denoised_img, f"{method_code}_denoised")
           
        
        return denoised_img
    
    def contrast_enhancement(self, read_img, method_code, saving=True):
        # CLAHE: Contrast Limited Adaptive Histogram Equalization
        if method_code.upper() == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_img = clahe.apply(read_img)
        # HE: Histogram Equalization  
        elif method_code.upper() == "HE":
            enhanced_img = cv2.equalizeHist(read_img)
        else:
            print("Please enter valid parameter(s)")
            return None
        if saving:
            self.saving_img(enhanced_img, f"{method_code}_enhanced")
        
        return enhanced_img
    # mean = std = (0.5, 0.5, 0.5)
    def zscore_norm(self, img, mean, std, saving=True):
        if mean and std:
            norm_img = (img - mean) / std
            if saving:
                self.saving_img(img, "zscore_normalized")
            return norm_img
        
            

class DATASET:
    def __init__(self, dataset_path):
        self.dataset_path=dataset_path

    def norm_parameters(self):
        if self.dataset_path:
            pixels_arr = []
            cls_folders = os.listdir(self.dataset_path)
            for cls_folder in cls_folders:
                images = os.listdir(os.path.join(self.dataset_path, cls_folder))                
                for img in images:
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        img_path = os.path.join(self.dataset_path, cls_folder, img)
                        read_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if read_img is not None:
                            pixels_arr.append(np.mean(read_img))
            
            if pixels_arr is not None:
                # pixels_arr = np.array(pixels_arr)
                mean = np.mean(pixels_arr)
                std = np.std(pixels_arr)
                return mean, std
        else: 
            print("There is no dataset path provided!")
            return None, None
        