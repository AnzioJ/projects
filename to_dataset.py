import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.transforms import functional as F

import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET

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
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all instances are of the class 'human'

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


# Example usage:
# Define the transformation
transform = F.to_tensor

# Create the dataset
dataset = CustomDataset(r'C:\Users\86183\OneDrive - The Ohio State University\Desktop\project1\yolo_person_train\images',
                        r'C:\Users\86183\OneDrive - The Ohio State University\Desktop\project1\yolo_person_train\xml_annotations', transform=transform)




