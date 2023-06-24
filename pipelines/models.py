"""
    Abstract base classes for detector models, tracker models, and the overall model.
"""

import sys
sys.path.append('..') # to import from parent directory

import numpy as np
import cv2
from abc import ABC, abstractmethod

from utils import drawing

class Object:
    """
        For keeping track of an object's history and current state.
    """
    def __init__(self, obj_id:int):
        self.obj_id = obj_id
        self.historical_boxes = []
        self.historical_confidences = []
        self.draw_color = np.random.randint(0, 255, size=(3,)).tolist()
        self.state = 'active'
        
    def add_xywh_box(self, box):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        self.historical_boxes.append([x1, y1, x2, y2])
    
    def add_xyxy_box(self, box):
        x1, y1, x2, y2 = box
        self.historical_boxes.append([x1, y1, x2, y2])
    
    def draw_current_box(self, frame):
        x1, y1, x2, y2 = self.historical_boxes[-1]
        drawing.draw_xyxy_box(frame, [x1, y1, x2, y2], self.draw_color)
        return frame

    def draw_historical_trace(self, frame, length=-1):
        points = len(self.historical_boxes)
        for i in range(points-1, max(points-length-1, -1)+1, -1):
            x1, y1, x2, y2 = self.historical_boxes[i]
            cx1, cy1 = int((x1 + x2) / 2), int((y1 + y2) / 2)
            x1, x2, x2, y2 = self.historical_boxes[i-1]
            cx2, cy2 = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.line(frame, (cx1, cy1), (cx2, cy2), self.draw_color, 2)
        return frame
    
    def draw_text(self, frame):
        x1, y1, x2, y2 = self.historical_boxes[-1]
        cv2.putText(frame, str(self.obj_id)+' '+self.state, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.draw_color, 2)
        return frame

    def draw(self, frame):
        frame = self.draw_current_box(frame)
        frame = self.draw_historical_trace(frame, length=30)
        frame = self.draw_text(frame)
        return frame
        

class Detector(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
class Tracker(ABC):
    @abstractmethod
    def __init__(self):
        pass

class Model(ABC):
    @abstractmethod
    def __init__(self):
        self.detector = None
        self.tracker = None
        pass
        
    @abstractmethod
    def update(self):
        pass