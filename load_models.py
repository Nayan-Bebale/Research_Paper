# tensorflow
import torchvision

import tensorflow_hub as hub

# for YOLO model
from ultralytics import YOLO

from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    KeypointRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights
)


# Function to load the specified model
def load_model(model_name="fasterrcnn"):
    if model_name == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "fasterrcnn_mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    elif model_name == "fasterrcnn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    elif model_name == "maskrcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "maskrcnn_v2":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    elif model_name == "keypointrcnn":
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "ssd":
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    elif model_name == "ssdlite":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()  # Set the model to evaluation mode
    return model



# Function to load TensorFlow models
def load_tf_model(model_name):
    if model_name == "ssd_mobilenet_v2":
        model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    elif model_name == "ssd_mobilenet_v1":
        model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1")
    elif model_name == "ssd_resnet50":
        model = hub.load("https://tfhub.dev/tensorflow/ssd_resnet50_v1_fpn_640x640/1")
        
    # Faster R-CNN models
    elif model_name == "faster_rcnn_resnet50":
        model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")
    elif model_name == "faster_rcnn_inception":
        model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
        
    # EfficientDet models
    elif model_name == "efficientdet_d0":
        model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")
    elif model_name == "efficientdet_d1":
        model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d1/1")
    elif model_name == "efficientdet_d2":
        model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d2/1")
    elif model_name == "efficientdet_d3":
        model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d3/1")
        
    # RetinaNet models
    elif model_name == "retinanet":
        model = hub.load("https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1")
        
    # CenterNet models
    elif model_name == "centernet_hourglass":
        model = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")
    elif model_name == "centernet_resnet50":
        model = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1")
        
    # Mask R-CNN models
    elif model_name == "mask_rcnn_resnet50":
        model = hub.load("https://tfhub.dev/tensorflow/mask_rcnn/resnet50_v1_fpn_1024x1024/1")
    elif model_name == "mask_rcnn_inception":
        model = hub.load("https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_atrous_coco/1")
    
    # Option for other models (e.g., YOLO, OpenPose, etc.)
    elif model_name == "yolo_v4":
        model = hub.load("https://tfhub.dev/tensorflow/yolov4-tiny/1")
    else:
        raise ValueError("Model name not recognized. Please choose a valid model.")
    
    return model

def load_yo_model(model_name):
    if model_name == "yolov5n":
        model = YOLO("yolov5s.pt")
    elif model_name == "yolov5s":
        model = YOLO('yolov5s.pt')
    elif model_name == "yolov5m":
        model = YOLO('yolov5m.pt')
    elif model_name == "yolov5l":
        model = YOLO('yolov5l.pt')
    elif model_name == "yolov5x":
        model = YOLO("yolov5x.pt")
    # yolo 8
    elif model_name == "yolov8n":
        model = YOLO("yolov8n.pt")
    elif model_name == "yolov8s":
        model = YOLO("yolov8s.pt")
    elif model_name == "yolov8m":
        model = YOLO("yolov8m.pt")
    elif model_name == "yolov8l":
        model = YOLO("yolov8l.pt")
    elif model_name == "yolov8x":
        model = YOLO("yolov8x.pt")
    # yolo 10
    elif model_name == "yolov10n":
        model = YOLO("yolov10n.pt")
    elif model_name == "yolov10s":
        model = YOLO("yolov10s.pt")
    elif model_name == "yolov10m":
        model = YOLO("yolov10m.pt")
    elif model_name == "yolov10l":
        model = YOLO("yolov10l.pt")
    elif model_name == "yolov10x":
        model = YOLO("yolov10x.pt")
        # Yolo 11
    elif model_name == "yolo11n":
        model = YOLO("yolo11n.pt")
    elif model_name == "yolo11s":
        model = YOLO("yolo11s.pt")
    elif model_name == "yolo11m":
        model = YOLO("yolo11m.pt")
    elif model_name == "yolo11l":
        model = YOLO("yolo11l.pt")
    elif model_name == "yolo11x":
        model = YOLO("yolo11x.pt")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

