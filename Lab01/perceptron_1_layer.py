import torch
from torch import nn
import torch.nn.functional as F

class Perceptron_1_Layer(nn.Module):
    def __init__ (self, image_size: tuple, num_labels: int):
        super().__init__()
        w,h = image_size
        self.linear = nn.Linear(
            in_features = w*h, 
            out_features = num_labels)
        
    def forward (self, x: torch.Tensor):
        bs = x.shape[0] # batch size
        x = x.reshape(bs, -1) # flatten ảnh 28x28 → vector 784
        x = self.linear(x) # nhân ma trận: [bs, 784] → [bs, 10]
        x = F.log_softmax(x, dim=1)
        return x