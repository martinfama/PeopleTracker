from parse import *

import time

from pipelines.csrt_full import CSRT
from pipelines.detectors import yolo, hog

# make a named and resizable window
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

# detector = yolo.YOLODetector()
detector = hog.HOGDetector()
# change startmode to 'select' if you want to select a bounding box
model = CSRT(startmode='detect', detector=detector)

logger.info('Starting to read video stream')
while 1:
    start_time = time.time()
    success, frame = cap.read()
    if frame is None: break
    model.update(frame)
    frame = model.draw(frame)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('img', frame)
    if out is not None: out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info('Quitting')
    
# clean
logger.info('Cleaning up')
cap.release()
del model
if out is not None: out.release()
logger.info('Done. Exiting.')