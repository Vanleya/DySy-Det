import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class CustomDataset(Dataset):
    def __init__(self, json_file, clip_path, img_size):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = CLIPProcessor.from_pretrained(clip_path)
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        label = self.data[idx]['label']
        img = Image.open(img_path).convert('RGB')
        img2 = img.resize(self.img_size)
        img_tensor = (transforms.PILToTensor()(img2) / 255.0 - 0.5) * 2
        inputs = self.processor(images=img, return_tensors="pt")
        clip_img_tensor = inputs['pixel_values'].squeeze()
        return img_tensor, clip_img_tensor, label