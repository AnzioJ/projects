import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import transforms as T
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm
from to_dataset import CustomDataset

# Define the transformations
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Function to create the model
def create_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

# Define a collate function for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

# Training function
def train_model():
    dataset = CustomDataset(r'C:\Users\86183\OneDrive - The Ohio State University\Desktop\project1\yolo_person_train\images',
                        r'C:\Users\86183\OneDrive - The Ohio State University\Desktop\project1\yolo_person_train\xml_annotations', transform=get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = create_model(num_classes=2)  # 1 class (human) + background
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            progress_bar.set_postfix({'loss': epoch_loss / len(data_loader)})
        
        lr_scheduler.step()

    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')

if __name__ == '__main__':
    train_model()


dataset = CustomDataset(r'C:\Users\86183\OneDrive - The Ohio State University\Desktop\project1\yolo_person_train\images',
                        r'C:\Users\86183\OneDrive - The Ohio State University\Desktop\project1\yolo_person_train\xml_annotations', transform=get_transform(train=True))
