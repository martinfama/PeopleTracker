from loguru import logger
logger.debug('Ready to debug')
logger.info('Starting to parse arguments (start of parse.py)')

import cv2
import argparse

# from hog_video import *
# from yolo_video import *
# from cv_track import *

from utils.dataloaders import ImgFolderCap

parser = argparse.ArgumentParser(
prog='PeopleTracker',
description='Track a single object, which is selected by the user, in a video file or webcam stream. Use different models.'
)

# we need the arguments:
# --videopath. The video to load. If not given, use webcam.
# --outpath. Where to save the output video. If not given, this is not saved.
# --model. Which model to use. If not given, use yolov3.

parser.add_argument('--videopath', help='path to video', default=0)
parser.add_argument('--outpath', help='path to output video', default=None)
parser.add_argument('--is_img_folder', help='whether the video is a folder of images', action='store_true')
parser.add_argument('--groundtruth', help='path to ground truth file', default=None)
parser.add_argument('--detector', help='detector to use', choices=['hog', 'yolo'] + cv_tracker_types, default='hog')
parser.add_argument('--tracker', help='tracker to use', choices=['CSRT', 'DeepSort', 'ByteTracker'], default='CSRT')

args = parser.parse_args()
logger.info('Parsed arguments: {}'.format(args))

# load the video
if args.is_img_folder:
    groundtruth_file = None
    if args.groundtruth is not None:
        groundtruth_file = args.groundtruth
    cap = ImgFolderCap(args.videopath, rectangle_ground_truth_file=groundtruth_file)
    logger.info('Using image folder: {}'.format(args.videopath))
else:
    cap = cv2.VideoCapture(args.videopath)
    if args.videopath == 0: logger.info('Using webcam')
    else: logger.info('Using video: {}'.format(args.videopath))

vid_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = cap.get(cv2.CAP_PROP_FPS)
logger.info('Video is of size: {} and has {} frames, at a framerate of {}'.format(vid_size, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps))

# output
out = None
if args.outpath:
    out = cv2.VideoWriter(args.outpath, cv2.VideoWriter_fourcc(*'XVID'), fps, vid_size)
logger.info('Will save output video to: {}'.format(args.outpath))

logger.info('Finished parsing arguments (end of parser.py)')