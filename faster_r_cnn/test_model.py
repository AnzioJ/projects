import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Function to create the model
def create_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

# Load the trained model
model = create_model(num_classes=2) 
model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth'))
model.eval()  # Set the model to evaluation mode

# Function to prepare the image
def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image).unsqueeze(0)  # Add batch dimension
    return image

# Prepare the test image
image_path = 'yolo_person_train\\test2.jpg'  # Replace with your image path
print("Loading image...")
image = prepare_image(image_path)

# Function to draw boxes on the image and check for detections
def draw_boxes_and_check_detection(image, prediction, threshold=0.28):
    image = F.to_pil_image(image.squeeze(0))  # Convert back to PIL image
    draw = ImageDraw.Draw(image)
    detected = False

    for element in range(len(prediction[0]['boxes'])):
        score = prediction[0]['scores'][element]
        if score > threshold:
            detected = True
            box = prediction[0]['boxes'][element].tolist()
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), text=f'{score:.2f}', fill="red")

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    return detected

# Perform object detection
with torch.no_grad():
    print("Running inference...")
    prediction = model(image)

# Check for detection and draw boxes
print("Drawing boxes...")
object_detected = draw_boxes_and_check_detection(image, prediction)
if object_detected:
    print("Objects detected.")
else:
    print("No objects detected.")
