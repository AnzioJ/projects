import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# 1. Base Convolutional Network
def create_base_network(input_shape):
    input_layer = Input(shape=input_shape)
    
    # Example architecture
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
    cls = Flatten()(cls)  # Flatten for classification

    # Regression Layer - Bounding Box Adjustment
    reg = Conv2D(num_anchors * 4, (1, 1), padding='valid')(shared_map)
    reg = Flatten()(reg)  # Flatten for regression

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

# Combine Components
input_shape = (256, 256, 3)
num_classes = 21  # Example number of classes (including background)
num_anchors = 9  # Example number of anchors

# Create Base Network
base_network = create_base_network(input_shape)

# Create RPN
feature_map_input = Input(shape=[None, None, 128])  # Output shape from the base network
rpn_model = create_rpn(feature_map_input, num_anchors)

# Create RoI Pooling Layer
roi_pooling_layer = ROIPoolingLayer(pooled_height=7, pooled_width=7)

# Create Classification and Regression Network
classification_and_regression_model = create_classification_and_regression_network((7, 7, 128), num_classes)

# Example Input for the Combined Model
image_input = Input(shape=input_shape)
feature_map = base_network(image_input)

# Note: In a full implementation, RPN outputs would be connected to RoI pooling, 
# and RoI pooling outputs would be connected to the classification and regression network.
# This requires additional logic to handle the dynamic nature of proposals.

# This setup shows the sequential flow of components.
# Further integration and training logic is needed for a functional model.

combined_model = Model(inputs=image_input, outputs=[feature_map, rpn_model.output, classification_and_regression_model.output])
