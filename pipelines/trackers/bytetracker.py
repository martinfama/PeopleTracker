import numpy as np
import cv2
import boxmot
from boxmot.utils.ops import xywh2xyxy

from ..models import Tracker

class ByteTracker(Tracker):
    def __init__(self):
        super().__init__()
        self.tracker = boxmot.BYTETracker(track_buffer=40)
        self.name = 'ByteTracker'
        self.unique_track_attrs = {}
        
    def update(self, detections, frame):
        # if type of detections is list, convert to numpy array
        if type(detections) == list:
            detections = np.array(detections)
            # reshape to always have 2 dimensions
            detections = detections.reshape(-1, 6)
        detections[:, 2:4] += detections[:, :2] # convert w,h columns to x2,y2 by adding x1,y1
        self.tracks = self.tracker.update(detections, frame)
        for track in self.tracks:
            x1, y1, x2, y2, track_id, _, cls = track.astype(int)
            # if track_id is not in unique_track_attrs, add it
            if track_id not in self.unique_track_attrs:
                self.unique_track_attrs[track_id] = {}
                # add a random color
                self.unique_track_attrs[track_id]['color'] = np.random.randint(0, 255, size=3).tolist()
    
    def get_tracks(self):
        track_boxes = []
        for lost_track in self.tracker.lost_stracks:
            x1, y1, x2, y2 = xywh2xyxy(np.expand_dims(lost_track.tlwh, axis=0)).astype(int).squeeze()
            track_boxes.append([x1, y1, x2, y2, lost_track.track_id, lost_track.score, lost_track.cls, 'lost'])
        
        for track in self.tracker.tracked_stracks:
            x1, y1, x2, y2 = xywh2xyxy(np.expand_dims(track.tlwh, axis=0)).astype(int).squeeze()
            track_boxes.append([x1, y1, x2, y2, track.track_id, track.score, track.cls, 'active'])        
        return track_boxes
    
    def draw_tracks(self, frame):
        track_boxes = []
        for removed_track in self.tracker.removed_stracks:
            x1, y1, x2, y2 = xywh2xyxy(np.expand_dims(removed_track.tlwh, axis=0)).astype(int).squeeze()
            #print('TRACKED_REMOVED: ', removed_track.track_id, ' : ', x1, y1, x2, y2)
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1-20), (x1+50, y1), color, -1)
            cv2.putText(frame, str(removed_track.track_id)+' REMOVED', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)
            
        for lost_track in self.tracker.lost_stracks:
            x1, y1, x2, y2 = xywh2xyxy(np.expand_dims(lost_track.tlwh, axis=0)).astype(int).squeeze()
            #print('TRACKED_LOST: ', lost_track.track_id, ' : ', x1, y1, x2, y2)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1-20), (x1+50, y1), color, -1)
            cv2.putText(frame, str(lost_track.track_id)+' LOST', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)
            track_boxes.append([x1, y1, x2, y2, lost_track.track_id, lost_track.score, lost_track.cls])
        
        for track in self.tracker.tracked_stracks:
            x1, y1, x2, y2 = xywh2xyxy(np.expand_dims(track.tlwh, axis=0)).astype(int).squeeze()
            #print('TRACKED_ACTIVE: ', track.track_id, ' : ', x1, y1, x2, y2)
            color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # draw solid color under text
            cv2.rectangle(frame, (x1, y1-20), (x1+50, y1), color, -1)
            cv2.putText(frame, str(track.track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)
            track_boxes.append([x1, y1, x2, y2, track.track_id, track.score, track.cls])
            
        return frame