import os
import sys

import cv2
from tqdm import tqdm

from torch_openpose import torch_openpose
from util import draw_bodypose

if __name__ == "__main__":
    tp = torch_openpose('body_25') # 默认使用 [0.7, 1.0, 1.3] 三个分辨率

    video_file="./videos/panoptic.mp4"
    cap = cv2.VideoCapture(video_file)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_size=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file.replace('.mp4','_processed.avi'),fourcc, 30.0, (int(width),int(height)),True)
    
    for index in tqdm(range(int(frame_size))):
        ret, frame = cap.read()
        if not ret:
            break
        poses = tp(frame)
        frame = draw_bodypose(frame, poses,'body_25')
        cv2.imwrite('./temp.jpg',frame)
        out.write(frame)
    
    cap.release()
    out.release()