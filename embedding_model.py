import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_gpu_utils import get_devices

class EmbeddingModel(nn.Module):
    def __init__(self, txt_features=100, img_features=2048, fc_features=2048, embed_dim=512):
        super().__init__()
        self.img_fc1 = nn.Linear(img_features, fc_features)
        self.img_fc2 = nn.Linear(fc_features, embed_dim)
        self.img_fc2_bn = nn.BatchNorm1d(embed_dim)
        self.txt_fc1 = nn.Linear(txt_features, fc_features)
        self.txt_fc2 = nn.Linear(fc_features, embed_dim)
        self.txt_fc2_bn = nn.BatchNorm1d(embed_dim)


    def forward(self, X_img, X_txt):
        X_img = F.relu(self.img_fc1(X_img))
        X_img = self.img_fc2_bn(self.img_fc2(X_img))
        Y_img = F.normalize(X_img)
        
        X_txt = F.relu(self.txt_fc1(X_txt))
        X_txt = self.txt_fc2_bn(self.txt_fc2(X_txt))
        Y_txt = F.normalize(X_txt)
        
        return Y_img, Y_txt
    
    
def create_models(gpus=1):
    resnet_model = models.resnet50(pretrained=True)
    resnet_model = nn.Sequential(*(list(resnet_model.children())[:-1]))
    device, gpu_indices = get_devices(gpus)
    resnet_model = nn.DataParallel(resnet_model, device_ids=gpu_indices)
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    
    similarity_model = EmbeddingModel()
    similarity_model = nn.DataParallel(similarity_model, device_ids=gpu_indices)
    similarity_model = similarity_model.to(device)

    return resnet_model, similarity_model, device    
