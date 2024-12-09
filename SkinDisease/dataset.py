import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DermNet(Dataset):
    def __init__(self):
        self.imgs_path = "DermNet/train/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            # Only take files with jpeg type
            for img_path in glob.glob(class_path + "/*.jpeg"):
                self.data.append([img_path, class_name])
            
        print(self.data[10])
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
        
        self.img_dim = (416, 416) # Resize to this dimension
        
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