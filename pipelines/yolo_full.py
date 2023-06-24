import sys
sys.path.append('..') # to import from parent directory

from loguru import logger
import cv2

from .models import *
from .detectors import yolo
from .trackers import bytetracker
from utils import object_selection

class YOLOModel(Model):
    def __init__(self, source='0', conf=0.8, iou=0.8, classes=[0]):
        self.detector = yolo.YOLODetector(conf_thres=conf, nms_thres=iou, classes=classes)
        logger.info(f'Detecting objects to track via {self.detector.name} detector')
        self.tracker = bytetracker.ByteTracker()
        logger.info(f'Initializing {self.tracker.name} tracker')
        self.tracked_objects = {}
        self.has_been_init = False
    
    def init_model(self, frame):
        tracks = self.tracker.get_tracks()
        selected_detections = object_selection.select_detections(frame, tracks, box_type='xyxy')
        if selected_detections == 'next':
            return
        for i, det in enumerate(selected_detections):
            box = det[:4]
            self.tracked_objects[i] = Object(obj_id=det[4])
            self.tracked_objects[i].add_xyxy_box(box)
            self.tracked_objects[i].state = det[-1]
        self.has_been_init = True
        
    def update(self, frame):
        detections = self.detector.detect(frame)
        self.tracker.update(detections, frame)
        tracks = self.tracker.get_tracks()
        if not self.has_been_init:
            if len(tracks) == 0:
                pass
            else:
                self.init_model(frame)
        else:
            for i, obj in self.tracked_objects.items():
                found = False
                for track in tracks:
                    if obj.obj_id == track[4]:
                        obj.add_xyxy_box(track[:4])
                        found = True
                        break
                if not found:
                    # track is essentially removed
                    obj.draw_color = (0, 0, 255)
                    obj.state = 'removed'
                
    def draw(self, frame):
        for i, obj in self.tracked_objects.items():
            obj.draw(frame)
        return frame