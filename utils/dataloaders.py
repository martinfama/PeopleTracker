"""
    Some auxiliary data loading utilities. Useful for example for loading
    an image folder, where the frames are saved as a sequence of images.
"""

import os, sys
import cv2
import numpy as np

class ImgFolderCap:
    """
        A class for loading a folder of images as a video. Tries to act like
        an OpenCV VideoCapture object as much as possible.
    """ 
    def __init__(self, dirpath, ext='.jpg', rectangle_ground_truth_file=None):
        self.dirpath = dirpath
        self.ext = ext
        self.files = []
        self.index = 0
        self.width = None
        self.height = None
        self.load_file_names()
        
        if rectangle_ground_truth_file is not None:
            self.rectangle_ground_truth_file = rectangle_ground_truth_file
            self.load_rectangle_ground_truth()
            
    def load_file_names(self):
        # load the file names, not the images themselves
        self.files = [f for f in os.listdir(self.dirpath) if f.endswith(self.ext)]
        self.files.sort()
    
    def load_rectangle_ground_truth(self):
        # load a sequence of rectangles that define the ground truth of the object to track
        self.rectangle_ground_truth = np.loadtxt(self.rectangle_ground_truth_file, delimiter=',')
    
    def read(self, update_index=True):
        # read the next frame
        if self.index >= len(self.files):
            return False, None
        filepath = os.path.join(self.dirpath, self.files[self.index])
        if update_index:
            self.index += 1
        frame = cv2.imread(filepath)
        if frame is None:
            return False, None
        return True, frame
    
    def read_rectangle_ground_truth(self):
        # read the next rectangle ground truth
        if self.index >= len(self.rectangle_ground_truth):
            return False, None
        rect = self.rectangle_ground_truth[self.index,:]
        return True, rect
    
    def get(self, prop):
        # for getting some properties as defined by OpenCV
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.files)
        
        if prop == cv2.CAP_PROP_FPS:
            return 30
        
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            if self.width is None:
                ret, frame = self.read(update_index=False)
                if ret:
                    self.width = frame.shape[1]
            return self.width
                
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            if self.height is None:
                ret, frame = self.read(update_index=False)
                if ret:
                    self.height = frame.shape[0]
            return self.height