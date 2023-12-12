import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
import torchvision.transforms as T
from to_dataset import CustomDataset  


# Define the transformations
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Function to create the model with an improved backbone
def create_model(num_classes, backbone_name='resnet101'):
    backbone = resnet_fpn_backbone(backbone_name, pretrained=True)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

# Define a collate function for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

# Training function with early stopping and visualization
def train_model():
    
    img_dir = 'yolo_person_train\images'
    annot_dir = 'yolo_person_train\\anotations'
    train_img_dir, train_annot_dir, val_img_dir, val_annot_dir = CustomDataset.split_dataset(img_dir, annot_dir)

    train_dataset = CustomDataset(train_img_dir, train_annot_dir, get_transform(train=True))
    val_dataset = CustomDataset(val_img_dir, val_annot_dir, get_transform(train=False))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = create_model(num_classes=2)  # Adjust num_classes as needed
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 20 
    best_loss = np.inf
    patience = 3
    patience_counter = 0

    epoch_train_loss = []
    epoch_val_loss = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, targets in tqdm(train_data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        train_loss /= len(train_data_loader)
        epoch_train_loss.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in tqdm(val_data_loader):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        val_loss /= len(val_data_loader)
        epoch_val_loss.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

        lr_scheduler.step()

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_train_loss, label='Train Loss')
    plt.plot(epoch_val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'fasterrcnn_resnet101_fpn.pth')

if __name__ == '__main__':
    train_model()
