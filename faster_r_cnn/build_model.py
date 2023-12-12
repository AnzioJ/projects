import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.model import Model

# 1. Base Convolutional Network
def create_base_network(input_shape):
    input_layer = Input(shape=input_shape)
    
    # Architecture
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    return Model(inputs=input_layer, outputs=x)

# 2. Region Proposal Network
def create_rpn(feature_map, num_anchors):
    """
    Create a Region Proposal Network (RPN).

    :param feature_map: A tensor representing the feature map from the base network.
    :param num_anchors: Number of anchors to consider at each location in the feature map.
    :return: A model outputting classification and regression predictions.
    """

    # Shared Convolutional Layer
    shared_map = Conv2D(512, (3, 3), padding='same', activation='relu')(feature_map)

    # Classification Layer - Object or Not
    cls = Conv2D(num_anchors * 2, (1, 1), padding='valid', activation='sigmoid')(shared_map)
    cls = Flatten()(cls)  

    # Regression Layer - Bounding Box Adjustment
    reg = Conv2D(num_anchors * 4, (1, 1), padding='valid')(shared_map)
    reg = Flatten()(reg)  

    # Create a Keras model
    model = tf.keras.Model(inputs=feature_map, outputs=[cls, reg])

    return model

# 3. RoI Pooling Layer
class ROIPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pooled_height, pooled_width):
        super(ROIPoolingLayer, self).__init__()
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

    def call(self, inputs):
        # Inputs: [feature_map, rois]
        feature_map = inputs[0]
        rois = inputs[1]

        outputs = []

        for roi in rois:
            x = roi[0]
            y = roi[1]
            w = roi[2]
            h = roi[3]

            x1 = int(x * feature_map.shape[2])
            y1 = int(y * feature_map.shape[1])
            x2 = int((x + w) * feature_map.shape[2])
            y2 = int((y + h) * feature_map.shape[1])

            roi_feature_map = feature_map[:, y1:y2, x1:x2, :]
            pooled_feature_map = tf.image.resize(roi_feature_map, (self.pooled_height, self.pooled_width))

            outputs.append(pooled_feature_map)

        final_output = tf.concat(outputs, axis=0)
        return final_output
    


# 4. Classification and Bounding Box Regression Network
def create_classification_and_regression_network(input_shape, num_classes):
    """
    Create a network for object classification and bounding box regression.

    :param input_shape: The shape of the input features (from RoI pooling).
    :param num_classes: The number of object classes (including background).
    :return: A Keras model for classification and bounding box regression.
    """

    # Input layer
    input_layer = tf.keras.Input(shape=input_shape)

    # Flatten the input
    x = Flatten()(input_layer)

    # Fully connected layers
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    # Classification layer
    classification_output = Dense(num_classes, activation='softmax', name='classification_output')(x)

    # Bounding box regression layer
    bbox_regression_output = Dense(num_classes * 4, activation='linear', name='bbox_regression_output')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=[classification_output, bbox_regression_output])

    return model

def apply_nms(predictions, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to prediction bounding boxes.

    :param predictions: A list of predictions, where each prediction is a tuple (box, score, class).
    :param iou_threshold: The Intersection over Union (IoU) threshold to use for determining overlap.
    :return: A list of filtered predictions after applying NMS.
    """
    filtered_predictions = []

    # Sort the predictions by score in descending order
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    while predictions:
        # Select the prediction with the highest score and remove it from the list
        best_prediction = predictions.pop(0)
        filtered_predictions.append(best_prediction)

        predictions = [pred for pred in predictions if pred[2] != best_prediction[2] or 
                       iou(best_prediction[0], pred[0]) < iou_threshold]

    return filtered_predictions

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    :param box1: Coordinates of the first box (x1, y1, x2, y2).
    :param box2: Coordinates of the second box (x1, y1, x2, y2).
    :return: IoU value.
    """
    # Calculate intersection area
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculate each box area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou

# Input Shape
input_shape = (256, 256, 3) 

# Number of anchors for RPN
num_anchors = 9

# Number of classes (including background)
num_classes = 2

# Create Base Network
base_network = create_base_network(input_shape)

# Feature Map Input for RPN 
feature_map_shape = base_network.output_shape[1:]
feature_map_input = tf.keras.Input(shape=feature_map_shape)

# Create RPN
rpn = create_rpn(feature_map_input, num_anchors)

# RoI Pooling Layer
roi_pooling_layer = ROIPoolingLayer(pooled_height=7, pooled_width=7)

# Classification and Regression Network
classification_and_regression_network = create_classification_and_regression_network((7, 7, feature_map_shape[-1]), num_classes)

# Model Input and Output
image_input = tf.keras.Input(shape=input_shape)

# Get feature map
feature_map = base_network(image_input)

# Get RPN outputs
rpn_cls, rpn_reg = rpn(feature_map)

# Placeholder for RoIs input
rois_input = tf.keras.Input(shape=(None, 4)) 

# RoI Pooling
pooled_features = roi_pooling_layer([feature_map, rois_input])

# Classification and Regression outputs
cls_output, reg_output = classification_and_regression_network(pooled_features)

# Create the model
faster_rcnn_model = tf.keras.Model(inputs=[image_input, rois_input], outputs=[rpn_cls, rpn_reg, cls_output, reg_output])

# Compile the model
faster_rcnn_model.compile(optimizer='adam', 
                          loss={'classification_output': 'categorical_crossentropy', 
                                'bbox_regression_output': 'mean_squared_error'},
                          metrics={'classification_output': 'accuracy'})
