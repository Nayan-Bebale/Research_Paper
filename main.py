import numpy as np
import subprocess
import os
import time
import torch
import tensorflow as tf
import transforms

import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from load_models import load_model, load_tf_model, load_yo_model

from sklearn.metrics import precision_recall_fscore_support

from parameters import (COCO_INSTANCE_CATEGORY_NAMES, get_image_size_in_bits,
                         calculate_latency, calculate_fps, calculate_iou,
                         calculate_energy, calculate_iou, calculate_map,
                         get_power_usage_nvidia, get_memory_utilization)


# Function to run the model and perform object detection
def run_tf_model(image_path, model_name, ground_truths):
    # Load the specified TensorFlow model
    model = load_tf_model(model_name)
    # Load and preprocess the image
    if image_path.startswith('http'):  # If the input is a URL
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    # Convert image to tensor and normalize
    image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension
    start_time = time.time()
    # Perform object detection
    detections = model(image_tensor)
    # Stop the inference time
    end_time = time.time()
    latency = calculate_latency(start_time, end_time)
    # Process the predictions
    detected_objects = []
    accuracy = 0.0  # Placeholder for accuracy calculation
    image_width, image_height = image.size
    for i in range(len(detections['detection_scores'][0])):
        score = detections['detection_scores'][0][i].numpy()
        if score > 0.5:  # Confidence threshold
            box = detections['detection_boxes'][0][i].numpy().tolist()  # Convert tensor to list
            label_id = int(detections['detection_classes'][0][i].numpy())  # Convert to Python int
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
            x_min = int(box[1] * image_width)
            y_min = int(box[0] * image_height)
            x_max = int(box[3] * image_width)
            y_max = int(box[2] * image_height)
            detected_objects.append({
                'label': label_name,
                'confidence': score,
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            })
            accuracy += score  # Sum the confidence for accuracy calculation
    # Calculate average accuracy
    accuracy /= len(detected_objects) if detected_objects else 1  # Avoid division by zero

    # Get dynamic power consumption from Nvidia GPU
    power_watts = get_power_usage_nvidia()

    # Calculate energy consumption in joules
    energy_required = calculate_energy(power_watts, latency)

        # Calculate metrics
    iou_scores = [calculate_iou(d['bbox'], gt['bbox']) for d, gt in zip(detected_objects, ground_truths or [])]
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    map_score = calculate_map(detected_objects, ground_truths or [])
    memory_utilization = get_memory_utilization()
    power_watts = get_power_usage_nvidia()
    energy_efficiency = calculate_energy(power_watts, latency)

    # F1-Score (if ground truths are provided)
    if ground_truths:
        y_true = [gt['label'] for gt in ground_truths]
        y_pred = [d['label'] for d in detected_objects]
        _, _, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    else:
        f1_score = 0
    

    # Get image size in bits
    image_size_bits = get_image_size_in_bits(image_path)
    throughput_result = float(f"{image_size_bits / latency if latency > 0 else 0:.2f}")
    throughput = throughput_result / 1000

    return (detected_objects, latency, accuracy, 
            throughput, energy_required, power_watts, 
            mean_iou, map_score, memory_utilization, energy_efficiency, f1_score)


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


def run_yo_model(image_path, model_name, ground_truths):
    model = load_yo_model(model_name)
    start_time = time.time()

    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    # Perform inference on the image
    results = model(image)

    end_time = time.time()
    # 
    latency = end_time - start_time

    detected_objects = []
    total_confidence = 0
    num_detections = 0

    for result in results:
        for box in result.boxes:
            obj_class = result.names[int(box.cls)]
            confidence = box.conf.item()
            bbox = box.xyxy.tolist()[0]  # x_min, y_min, x_max, y_max

            # Transform bbox into the required format
            x_min, y_min, x_max, y_max = bbox
            detected_objects.append({
                'label': obj_class,
                'confidence': confidence,
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            })
            total_confidence += confidence
            num_detections += 1

    # Calculate accuracy
    overall_accuracy = total_confidence / num_detections if num_detections > 0 else 0

        # Calculate metrics
    iou_scores = [calculate_iou(d['bbox'], gt['bbox']) for d, gt in zip(detected_objects, ground_truths or [])]
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    map_score = calculate_map(detected_objects, ground_truths or [])
    memory_utilization = get_memory_utilization()
    power_watts = get_power_usage_nvidia()
    energy_efficiency = calculate_energy(power_watts, latency)

    # F1-Score (if ground truths are provided)
    if ground_truths:
        y_true = [gt['label'] for gt in ground_truths]
        y_pred = [d['label'] for d in detected_objects]
        _, _, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    else:
        f1_score = 0

    # Get dynamic power consumption from Nvidia GPU
    power_watts = get_power_usage_nvidia()

    # Calculate energy consumption in joules
    energy_required = calculate_energy(power_watts, latency)

    # Get image size in bits
    image_size_bits = get_image_size_in_bits(image_path)
    throughput_result = float(f"{image_size_bits / latency if latency > 0 else 0:.2f}")
    throughput = throughput_result / 1000

    return (detected_objects, latency, overall_accuracy, 
            throughput, energy_required, power_watts, 
            mean_iou, map_score, memory_utilization, energy_efficiency, f1_score)



# Function to run the model and perform object detection
def run_model(image_path, ground_truths, model_name="fasterrcnn"):
    if model_name[:2] == 'tf':
        model_ = model_name[3::]
        return run_tf_model(image_path=image_path,model_name=model_)
    elif model_name[:2] == 'yo':
        model_ = model_name[3::]
        return run_yo_model(image_path=image_path,model_name=model_)

    # Load the specified model
    model = load_model(model_name)

    # Load and preprocess the image
    if image_path.startswith('http'):  # If the input is a URL
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print(image_tensor.shape)
    start_time = time.time()
    
    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Stop the inference time
    end_time = time.time()
    latency = calculate_latency(start_time, end_time)

    # Process the predictions
    detected_objects = []
    accuracy = 0.0  # Placeholder for accuracy calculation

    if 'scores' in predictions[0] and 'labels' in predictions[0] and 'boxes' in predictions[0]:
        for i, score in enumerate(predictions[0]['scores'].tolist()):  # Convert to list for compatibility
            if score > 0.5:  # Confidence threshold
                box = predictions[0]['boxes'][i].tolist()  # Convert tensor to list
                label_id = predictions[0]['labels'][i].item()  # Convert to Python int
                
                if type(label_id) == int:
                    # Convert label_id to string using the COCO class names
                    label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]  # Convert to string class name
                    # print(f"Detected object label: {label_name} (confidence: {score})")
                else:
                    label_name = label_id
                    
                detected_objects.append({
                    'label': label_name,  # Store the string label name
                    'confidence': score,
                    'x': box[0],  # x_min
                    'y': box[1],  # y_min
                    'width': box[2] - box[0],  # width
                    'height': box[3] - box[1]  # height
                })
                accuracy += score  # Sum the confidence for accuracy calculation

    # Calculate average accuracy
    accuracy /= len(detected_objects) if detected_objects else 1  # Avoid division by zero
    # Get dynamic power consumption from Nvidia GPU
    power_watts = get_power_usage_nvidia()

    # Calculate metrics
    iou_scores = [calculate_iou(d['bbox'], gt['bbox']) for d, gt in zip(detected_objects, ground_truths or [])]
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    map_score = calculate_map(detected_objects, ground_truths or [])
    memory_utilization = get_memory_utilization()
    power_watts = get_power_usage_nvidia()
    energy_efficiency = calculate_energy(power_watts, latency)

    # F1-Score (if ground truths are provided)
    if ground_truths:
        y_true = [gt['label'] for gt in ground_truths]
        y_pred = [d['label'] for d in detected_objects]
        _, _, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    else:
        f1_score = 0


    # Calculate energy consumption in joules
    energy_required = calculate_energy(power_watts, latency)

    # Get image size in bits
    image_size_bits = get_image_size_in_bits(image_path)

    throughput_result = float(f"{image_size_bits / latency if latency > 0 else 0:.2f}")
    throughput = throughput_result / 1000

    return (detected_objects, latency, accuracy, 
            throughput, energy_required, power_watts, 
            mean_iou, map_score, memory_utilization, energy_efficiency, f1_score)