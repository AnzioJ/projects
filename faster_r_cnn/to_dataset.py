import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.transforms import functional as F
import random

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotations_dir, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.annotations = list(sorted(os.listdir(annotations_dir)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        anno_path = os.path.join(self.annotations_dir, self.annotations[idx])
        tree = ET.parse(anno_path)
        root = tree.getroot()

        boxes = []
        for member in root.findall('object'):
            xmin = int(member.find('bndbox/xmin').text)
            ymin = int(member.find('bndbox/ymin').text)
            xmax = int(member.find('bndbox/xmax').text)
            ymax = int(member.find('bndbox/ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64) 

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    def split_dataset(img_dir, annot_dir, train_size=0.8):
        # List all images
        images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        random.shuffle(images)  # Shuffle the dataset

        # Split the dataset
        train_count = int(len(images) * train_size)
        train_images = images[:train_count]
        val_images = images[train_count:]

        # Create training and validation directories
        train_img_dir = os.path.join(img_dir, 'train')
        val_img_dir = os.path.join(img_dir, 'val')
        train_annot_dir = os.path.join(annot_dir, 'train')
        val_annot_dir = os.path.join(annot_dir, 'val')

        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_annot_dir, exist_ok=True)
        os.makedirs(val_annot_dir, exist_ok=True)

        # Move files to their respective directories
        for img_file in train_images:
            os.rename(os.path.join(img_dir, img_file), os.path.join(train_img_dir, img_file))
            os.rename(os.path.join(annot_dir, img_file.replace('.jpg', '.xml')), os.path.join(train_annot_dir, img_file.replace('.jpg', '.xml')))
        
        for img_file in val_images:
            os.rename(os.path.join(img_dir, img_file), os.path.join(val_img_dir, img_file))
            os.rename(os.path.join(annot_dir, img_file.replace('.jpg', '.xml')), os.path.join(val_annot_dir, img_file.replace('.jpg', '.xml')))

        return train_img_dir, train_annot_dir, val_img_dir, val_annot_dir



# Define the transformation
transform = F.to_tensor

# Create the dataset
dataset = CustomDataset(r'C:\Users\86183\OneDrive - The Ohio State University\Desktop\project1\yolo_person_train\images',
                        r'C:\Users\86183\OneDrive - The Ohio State University\Desktop\project1\yolo_person_train\xml_annotations', transform=transform)





