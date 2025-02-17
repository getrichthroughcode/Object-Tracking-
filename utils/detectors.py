"""
Author: Abdoulaye DIALLO <abdoulayediallo338@gmail.com>
This file contains the implementation of different detectors
for the purpose of tracking objects in a video.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", use_trt=False):
        """
        Initializes the YOLO detector.
        :param model_path: Path to the YOLO model or model name.
        :param use_trt: Whether to use TensorRT optimization.
        """
        self.model_path = model_path
        self.use_trt = use_trt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        if use_trt:
            self.model.export(format="engine")  
            self.model_path = model_path.replace(".pt", ".engine")
            print(f"Model converted to TensorRT: {self.model_path}")

    def detect(self, image):
        """
        Perform object detection on an image.
        :param image: Input image in BGR format.
        :return: Processed detections.
        """
        results = self.model(image)  # Peut être un seul objet ou une liste
        """
        cls, confs, xywh = [], [], []
        
        # Vérifier si results est une liste (plusieurs images) ou un seul objet
        if isinstance(results, list):
            for result in results:  # Parcours des résultats pour chaque image
                cls.append(result.boxes.cls.tolist())
                confs.append(result.boxes.conf.tolist())
                xywh.append(result.boxes.xywh.tolist())
        else:
        """    
        cls = results[0].boxes.cls.tolist()
        confs = results[0].boxes.conf.tolist()
        xywh = results[0].boxes.xywh.tolist()

        return cls, confs, xywh

    def process_results(self, results, image):
        """
        Draw bounding boxes on the image.
        """
        #print(f"{type(results)=}")
        for result in results:
            #print(f"{result=}")
            #print(f"{result.boxes=}")
            #print(f"{result.boxes.xyxy=}")
            
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])  # Get bounding box coordinates
            #print(f"{(x1, y1, x2, y2)=}")
            """
            conf = result.conf[0].item()
            cls = int(result.cls[0])
            
            label = f"Class {cls}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            """
            #result.show()
        return image

