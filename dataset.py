from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class BasicDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.image_ids = os.listdir(images_dir)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_ids[idx])
        image = Image.open(img_path)
        label = self.images_dir.split('/')[-1]
        if self.transform:
            image = self.transform(image)
        return image, label, self.image_ids[idx]
    