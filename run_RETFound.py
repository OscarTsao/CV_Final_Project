import kagglehub

# Download latest version
path = kagglehub.dataset_download("ninadaithal/imagesoasis")

print("Path to dataset files:", path)

import torch
import os
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image

# import dataset
from dataset import BasicDataset
from torch.utils.data import DataLoader, ConcatDataset

data_path = "C://Users/Lab308/.cache/kagglehub/datasets/ninadaithal/imagesoasis/versions/1/Data"
non_demented_path = data_path + "/Non Demented"
mild_demented_path = data_path + "/Mild Dementia"
moderate_demented_path = data_path + "/Moderate Dementia"
very_mild_demented_path = data_path + "/Very mild Dementia"

# Load the dataset
non_dataset = BasicDataset(non_demented_path)
mild_dataset = BasicDataset(mild_demented_path)
moderate_dataset = BasicDataset(moderate_demented_path)
very_mild_dataset = BasicDataset(very_mild_demented_path)
dataset = ConcatDataset([non_dataset, mild_dataset, moderate_dataset, very_mild_dataset])
Alldata = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

#Split the dataset
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)

# Model settings
from Model.RETFound.models_mae import MaskedAutoencoderViT
from Model.RETFound.models_vit import VisionTransformer
from timm.optim import optim_factory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_mae = MaskedAutoencoderViT(embed_dim=768, depth=12, num_heads=12)

model_autoencoder = VisionTransformer(embed_dim=786, depth=12, num_heads=12, num_classes=4)

param_groups = optim_factory.add_weight_decay(model_mae, 0.05)
optimizer = torch.optim.AdamW(param_groups, lr=1.5e-4, betas=(0.9, 0.95))

model_mae = model_mae.to(device)
model_autoencoder = model_autoencoder.to(device)

def load_checkpoints(epoch, model, optimizer, stage):
    checkpoint_path = f"checkpoints/ConvNeXtV2/{stage}"
    
    if os.path.exists(checkpoint_path):
        print(f"Load checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")

def save_checkpoints(epoch, model, optimizer, stage):
    checkpoint_path = f"checkpoints/ConvNeXtV2/{stage}"
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, checkpoint_path + f"/checkpoint_{epoch}.pth")
    
    
def pretrain():
    pretrain_epoch = 2000
    start_epoch = 0

    epoch_loss = 0

    if start_epoch:
        load_checkpoints(start_epoch-1, model_mae, optimizer, stage="pretrain")

    for epoch in range(start_epoch, pretrain_epoch):
        torch.cuda.empty_cache()

        localtime = time.asctime( time.localtime(time.time()) )
        tqdm.write('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,pretrain_epoch,localtime))
        tqdm.write('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,pretrain_epoch, localtime)))

        folder_name = os.path.join("see_image", "pre-train")
        os.makedirs(folder_name, exist_ok=True)

        torch.cuda.empty_cache()
        
        for idx, batch in tqdm(enumerate(train_loader)):
            img, label, id = batch
            img = img.to(device)


            for i in range(int(img.size(0))):
                folder_target = os.path.join(folder_name, "target")
                os.makedirs(folder_target, exist_ok=True)
                image = img[i].detach().cpu().numpy().transpose(1, 2, 0)
                image = np.clip(image * 255, 0, 255)
                image = Image.fromarray(image.astype(np.uint8))
                image.save(os.path.join(folder_target, str(id[i]) + ".png"))
                
                model_mae.train()
                model_autoencoder.eval()
                optimizer.zero_grad()

                loss, pred, mask = model_mae(imgs=img)
                
            for i in range(int(pred.size(0))):
                folder_t1_prob = os.path.join(folder_name, "mae")
                os.makedirs(folder_t1_prob, exist_ok=True)
                image_prob = pred[i].detach().cpu().numpy().transpose(1, 2, 0)
                image_prob = Image.fromarray(image_prob.astype(np.uint8))
                image_prob.save(os.path.join(folder_t1_prob, str(id[i])))
                
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            
            print(f'Epoch{epoch+1} loss : \n pretrain loss : {epoch_loss}')

            if (epoch+1) % 10 == 0:
                save_checkpoints(epoch+1, warm_up=True)
        
        
if __name__ == '__main__':
    pretrain()