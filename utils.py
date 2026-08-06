import cv2
from ultralytics import YOLO, RTDETR
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

# ==========================================
# تحميل النماذج
# ==========================================

@st.cache_resource
def load_detection_model(model_name):
    if model_name == "YOLO":
        return YOLO('models/best_yolo.pt') 
    elif model_name == "RT-DETR":
        return RTDETR('models/best_rtdetr.pt')
    return None

@st.cache_resource
def load_classification_model(path, num_classes=91):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading ResNet weights: {e}")
        
    model.to(device)
    model.eval()
    return model, device

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_fast_pipeline(det_model, clf_model, device, image, conf_threshold, class_names):
   
    det_results = det_model.predict(image, conf=conf_threshold, verbose=False)
    boxes = det_results[0].boxes
    
    plotted_image = det_results[0].plot(labels=False)    
    if len(boxes) == 0:
        return plotted_image, []

    img_rgb = image.convert('RGB')
    w, h = image.size
    
    crops_tensors = []
    metadata = [] 
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        cropped = img_rgb.crop((x1, y1, x2, y2))
        
        if cropped.size[0] < 5 or cropped.size[1] < 5:
            continue
            
        crops_tensors.append(resnet_transform(cropped))
        metadata.append({
            "bbox": [x1, y1, x2, y2],
            "det_conf": box.conf.item() * 100
        })

    if not crops_tensors:
        return plotted_image, []

    batch_tensor = torch.stack(crops_tensors).to(device)
    
    final_results = []
    batch_size = 32 
    with torch.no_grad():
        for i in range(0, len(batch_tensor), batch_size):
            batch = batch_tensor[i : i + batch_size]
            outputs = clf_model(batch)
            probs = F.softmax(outputs, dim=1)
            
            top_probs, top_ids = torch.topk(probs, 1)
            
            for j in range(len(batch)):
                idx = i + j
                class_id = top_ids[j].item()
                conf = top_probs[j].item() * 100
                
                label = class_names[class_id] if class_id < len(class_names) else f"Unknown"
                
                final_results.append({
                    "bbox": metadata[idx]["bbox"],
                    "label": label,
                    "confidence": conf,
                    "det_conf": metadata[idx]["det_conf"]
                })
                
    return plotted_image, final_results