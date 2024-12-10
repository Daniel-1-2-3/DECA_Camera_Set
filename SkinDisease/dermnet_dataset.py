import glob
import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class DermNet(Dataset):
    def __init__(self):
        self.imgs_path = 'C:\\Daniel\\High School\\Python\\Disease_Classification\\SkinDisease\\DermNet\\train\\*'
        file_list = glob.glob(self.imgs_path) 
        self.data = []
        
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            # Only take files with jpg type
            for img_path in glob.glob(class_path + "\\*.jpg"):
                self.data.append([img_path, class_name])
    
        self.class_map = {
            "Acne": 0,
            "Atopic Dermatitis": 1,
            "Bacterial Infection": 2, 
            "Benign Tumor": 3,
            "Bullous Disease": 4,
            "Eczema": 5,
            "Lupus": 6,
            "Lyme Disease": 7,
            "Malignant Lesions": 8, 
            "Mole": 9,
            "Nail Fungus": 10,
            "Poison Ivy": 11,
            "STD": 12,
            "Viral Infection": 13
        }
        
        self.img_dim = (512, 512) # Resize to this dimension
        print("Sample Data:", self.data[0:3])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

if __name__ == "__name__":
    dataset = DermNet()