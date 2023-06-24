import cv2

def resize_video(fname, ratio, outname):
    # load fname, and resize by ratio
    cap = cv2.VideoCapture(fname)
    vid_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(outname, cv2.VideoWriter_fourcc(*'XVID'), fps, (int(vid_size[0]*ratio), int(vid_size[1]*ratio)))
    while 1:
        ret, frame = cap.read()
        if not ret: break
        out.write(cv2.resize(frame, (int(vid_size[0]*ratio), int(vid_size[1]*ratio))))
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Resize a video')
    parser.add_argument('--fname', type=str, help='path to video')
    parser.add_argument('--ratio', type=float, help='ratio to resize by')
    parser.add_argument('--outname', type=str, help='path to output video')
    args = parser.parse_args()
    resize_video(args.fname, args.ratio, args.outname)