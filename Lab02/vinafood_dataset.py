import torch
from torch.utils.data import Dataset
import os
import cv2 

def collate_fn(samples: list[dict]):
    #images = [sample['image'].permute(1, 2, -1).unsqueeze(0) for sample in samples] 
    images = [sample['image'] for sample in samples]
    labels = [sample['label'] for sample in samples] 
    
    images = torch.cat(images, dim=0)
    labels = torch.tensor(labels)
    
    return {
        'image': images,
        'label': labels
    }

class VinaFood(Dataset):
    #def __init__(self, path: str):
    def __init__(self, path: str, image_size: tuple[int]):
        super().__init__()
    
        self.image_size = image_size
        self.label2idx = {}
        self.idx2label = {}
        self.data: list[dict] = self.load_data(path)
        
    def load_data(self, path):
        data = []
        label_id = 0
        for folder in os.listdir(path):
            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_id
                label_id += 1
            for image_file in os.listdir(os.path.join(path, folder)):
                image = cv2.imread(os.path.join(path, folder, image_file))
                data.append({
                    'image': image,
                    'label': label
                })
        self.idx2label = {id: label for label, id in self.label2idx.items()}
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        image = item['image']
        label = item['label']
        
        # image = cv2.resize(image, (224, 224))
        # label_id = self.label2idx[label]
        
        image = cv2.resize(image, self.image_size)
        image = torch.tensor(image)
        image = image.permute(-1, 0, 1)
        
        return {
            'image': image,
            #'label': label_id
            'label': self.label2idx[label]
        }
    
    
    