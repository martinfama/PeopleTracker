from loguru import logger

import numpy as np
import cv2

from . import drawing

mouse_pos = None
clicked = False
def mouse_callback(event, x, y, flags, param):
    global mouse_pos, clicked
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
    
def select_detections(frame, detections, windowName='img', box_type='xyxy'):
    """
        Function to select detections from a frame. Returns a list of selected detections.
        You can select multiple detections by clicking on them, and deselect them by clicking again.
        Press n to skip to the next frame, and q to quit and return the selected detections.

        Args:
            frame (np.array): Frame to select detections from.
            detections (list): List of detections to select from.
            windowName (str, optional): The cv2 window name. Defaults to 'img'.
            box_type (str, optional): Are boxes given in xyxy format or xywh. Defaults to 'xyxy'.

        Returns:
            list: List of selected detections.
    """
    logger.info('Select detections to track. Press q to quit, press n to skip to next frame.')
    draw_func = drawing.draw_xyxy_boxes if box_type == 'xyxy' else drawing.draw_xywh_boxes
    
    if np.array(detections).ndim == 1:
        detections = [detections]
    selection_boxes = np.array(detections)[:, :4].astype(int)
    
    selected = []
    for box in selection_boxes:
        selected.append(False)
    colors = [(0, 255, 0)  if s else (255, 0, 0) for s in selected]
    draw_func(frame, selection_boxes, colors)
    
    global mouse_pos, clicked
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(windowName, mouse_callback)
    while 1:
        # if mouse is clicked, loop over boxes
        # if mouse is clicked inside a box, select that box
        if clicked:
            for i, box in enumerate(selection_boxes):
                if box_type == 'xyxy':
                    x1, y1, x2, y2 = box
                else:
                    x1, y1, w, h = box
                    x2, y2 = x1 + w, y1 + h
                if x1 < mouse_pos[0] < x2 and y1 < mouse_pos[1] < y2:
                    selected[i] = not selected[i]
                    colors[i] = (0, 255, 0) if selected[i] else (255, 0, 0)
            clicked = False
        
        # draw boxes
        draw_func(frame, selection_boxes, colors)
        cv2.imshow(windowName, frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.waitKey(1) & 0xFF == ord('n'): return "next"
    
    selected_detections = [detection for i, detection in enumerate(detections) if selected[i]]
    logger.info(f'Select a total of {len(selected_detections)} detections.')
    for s_det in selected_detections:
        logger.info(f'Selected detection: {s_det}')
    
    return selected_detections