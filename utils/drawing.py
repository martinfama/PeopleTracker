import cv2

def draw_xyxy_box(frame, box, color=(0,255,0)):
    x1, y1, x2, y2 = box
    p1 = int(x1), int(y1)
    p2 = int(x2), int(y2)
    cv2.rectangle(frame, p1, p2, color, 2)

def draw_xywh_box(frame, box, color=(0,255,0)):
    x1, y1, w, h = box
    p1 = int(x1), int(y1)
    p2 = int(x1 + w), int(y1 + h)
    cv2.rectangle(frame, p1, p2, color, 2)

def draw_xyxy_boxes(frame, boxes, colors):
    for i, box in enumerate(boxes):
        draw_xyxy_box(frame, box, colors[i])
        
def draw_xywh_boxes(frame, boxes, colors):
    for i, box in enumerate(boxes):
        draw_xywh_box(frame, box, colors[i])