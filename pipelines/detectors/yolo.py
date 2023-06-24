from ..models import Detector

import torch

from ultralytics import YOLO

class YOLODetector(Detector):
    """
        Defines a YOLO detector class that inherits from the Detector class.
        Uses YOLO model from ultralytics.
        Returns detections in the format [x1, y1, w, h, conf, cls]
    """                                                                                    # class 0 corresponds to 'person'
    def __init__(self, model_path='model_files/yolov8n.pt', conf_thres=0.6, nms_thres=0.2, classes=[0]):
        super().__init__()
        self.model = YOLO(model_path, )
        self.name = 'yolo'
        self.classes = classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        
        self.detections = []
    
    def detect(self, frame):
        self.model.to(self.device)
        detections = self.model(frame, conf=self.conf_thres, iou=self.nms_thres, classes=self.classes, verbose=False)
        self.detections = []
        for x1, y1, x2, y2, conf, cls in detections[0].boxes.data.detach().numpy():
            self.detections.append( [x1, y1, x2-x1, y2-y1, conf, int(cls)] )
        return self.detections