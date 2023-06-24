import cv2
import numpy as np
from ..models import Detector

class HOGDetector(Detector):
    """
        Defines a HOG detector class that inherits from the Detector class.
        Uses HOG model from OpenCV, specifically the HOGDescriptor class for people.
        Returns detections in the format [x1, y1, w, h, weight]
    """
    def __init__(self, winStride=(16,16)):
        super().__init__()
        self.name = 'hog'
        self.model = cv2.HOGDescriptor()
        self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.winStride = winStride
        self.detections = []
        
    def detect(self, frame):
        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = self.model.detectMultiScale(frame, winStride=self.winStride)
        # boxes is a 4xn array, and weights is a n array. add column of weights to boxes
        detections = np.c_[boxes, weights]
        self.detections = detections
        return detections