import sys
sys.path.append('..') # to import from parent directory

from loguru import logger
import cv2

from .models import *
from utils import object_selection
from utils import drawing

class CSRT(Model):
    def __init__(self, startmode='select', **kwargs):
        logger.info('Initializing CSRT model. Startmode: {}', startmode)
        self.model = cv2.legacy.MultiTracker_create()
        self.tracked_objects = {}
        self.has_been_init = False
        self.startmode = startmode
        if self.startmode == 'detect':
            self.detector = kwargs['detector']
    
    def init_model(self, frame):
        if self.startmode == 'detect':
            logger.info(f'Detecting objects to track via {self.detector.name} detector')
            detections = self.detector.detect(frame)
            if len(detections) == 0:
                logger.warning('No objects detected. Defaulting to ROI selection')
                self.startmode = 'select'
            else:
                boxes = object_selection.select_detections(frame, detections, box_type='xywh')
        if self.startmode == 'select':
            logger.info('Selecting objects to track via ROI selection')
            boxes = cv2.selectROIs('img', frame, False)
            
        logger.info(f'Initializing {len(boxes)} CSRT trackers')
        for i, box in enumerate(boxes):
            box = box[:4]
            tracker = cv2.legacy.TrackerCSRT_create()
            self.model.add(tracker, frame, box)
            self.tracked_objects[i] = Object(obj_id=i)
            self.tracked_objects[i].add_xywh_box(box)
        self.has_been_init = True
        logger.info('CSRT model initialized')

    def update(self, frame):
        if not self.has_been_init:
            self.init_model(frame)
        success, boxes = self.model.update(frame)
        for i, box in enumerate(boxes):
            self.tracked_objects[i].add_xywh_box(box)
    
    def draw(self, frame):
        for i, obj in self.tracked_objects.items():
            obj.draw_current_box(frame)
            obj.draw_historical_trace(frame, length=40)
        return frame