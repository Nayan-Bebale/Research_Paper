import os
import numpy as np
import subprocess
import psutil


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "TV",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Function to get real-time power consumption (for Nvidia GPUs)
def get_power_usage_nvidia():
    try:
        # Execute nvidia-smi to get power consumption
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
        power_watts = float(output.decode('utf-8').strip())
        return power_watts
    except Exception as e:
        print(f"Error fetching power consumption: {e}")
        return 0  # Return 0 if unable to fetch
    


# Function to calculate energy required
def calculate_energy(power_watts, latency_seconds):
    return power_watts * latency_seconds  # Energy in joules (watts * seconds)

# Function to calculate latency and response time
def calculate_latency(start_time, end_time):
    return end_time - start_time

# Function to calculate image size in bits
def get_image_size_in_bits(image_path):
    file_size_bytes = os.path.getsize(image_path)  # File size in bytes
    file_size_bits = file_size_bytes * 8  # Convert to bits (1 byte = 8 bits)
    return file_size_bits

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0

# Function to calculate mean average precision (mAP)
def calculate_map(detections, ground_truths, iou_threshold=0.5):
    average_precisions = []
    for label in set(gt['label'] for gt in ground_truths):
        preds = [d for d in detections if d['label'] == label]
        truths = [gt for gt in ground_truths if gt['label'] == label]
        tp, fp, scores = [], [], []

        for pred in preds:
            scores.append(pred['confidence'])
            max_iou = 0
            for truth in truths:
                iou = calculate_iou(pred['bbox'], truth['bbox'])
                max_iou = max(max_iou, iou)
            if max_iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        # Compute precision-recall for this label
        precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        recall = np.cumsum(tp) / len(truths)
        average_precisions.append(np.mean(precision))

    return np.mean(average_precisions) if average_precisions else 0

# Function to measure memory utilization
def get_memory_utilization():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  

# Function to calculate FPS (Inference Speed)
def calculate_fps(latency):
    return 1 / latency if latency > 0 else 0
