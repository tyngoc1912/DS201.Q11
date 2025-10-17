import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np

# def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
#     # items = [{
#     #     "image": items["image"].expand_dims(items["image"], axis=0),
#     #     "label": np.array(items["label"])
#     # } for item in items]
    
#     data = {
#         "image" : np.stack([item["image"] for item in items], axis=0),
#         "label" : np.stack([item["label"] for item in items], axis=0)
#     }
    
#     data = {
#         "image": torch.tensor(data["image"]),
#         "label": torch.tensor(data["label"])
#     }
#     return data
    
def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
    images = np.stack(
        [np.expand_dims(item["image"], axis=0) for item in items],
        axis=0
    ).astype(np.float32) / 255.0

    labels = np.stack([item["label"] for item in items], axis=0).astype(np.int64)

    data = {
        "image": torch.from_numpy(images).float(),  # ép về float32
        "label": torch.from_numpy(labels).long()    # ép về int64
    }
    return data    
    

class Items:
    def __init__(self, image, label):
        self.image = image
        self.label = label
        
class MNISTDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)
        
        self._data = [
            {
                "image": np.array(image),
                "label": label
            }
            for image, label in zip(images.tolist(), labels.tolist())
        ]
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]