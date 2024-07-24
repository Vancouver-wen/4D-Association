import os
import sys
import re
if __name__=="__main__":
    sys.path.append("./")

import numpy as np
import cv2
from tqdm import tqdm

from association.openpose_detection import SKEL19DEF,SKEL25DEF,OpenposeDetection

def read_openpose_detection(txt_path):
    with open(txt_path,'r') as f:
        string=f.read()
    numbers = re.findall(r"\d+\.?\d*",string) # 提取所有的数字, 包括 整数与小数
    numbers=[float(number) if '.' in number else int(number) for number in numbers]
    numbers=list(reversed(numbers)) # pop(-1)的时间复杂度远低于 pop(0)
    skel_type=numbers.pop(-1)
    frame_size=numbers.pop(-1)
    joint_size=SKEL25DEF.joint_size
    paf_size=SKEL25DEF.paf_size
    paf_dict=SKEL25DEF.paf_dict
    openposeDetections=[]
    for _ in range(frame_size):
        openposeDetection=OpenposeDetection()
        # read joints
        openposeDetection.joints=[]
        for _ in range(joint_size):
            joints=[]
            joint_num=numbers.pop(-1)
            assert isinstance(joint_num,int)
            if joint_num==0:
                openposeDetection.joints.append(joints)
                continue
            xs=[]
            for _ in range(joint_num):
                xs.append(numbers.pop(-1))
            ys=[]
            for _ in range(joint_num):
                ys.append(numbers.pop(-1))
            confs=[]
            for _ in range(joint_num):
                confs.append(numbers.pop(-1))
            for x,y,conf in list(zip(xs,ys,confs)):
                joints.append([x,y,conf])
            openposeDetection.joints.append(joints)
        # read pafs
        openposeDetection.pafs=[]
        for paf_pair in paf_dict:
            ja,jb=paf_pair
            paf=[]
            for _ in openposeDetection.joints[ja]:
                each_paf=[]
                for _ in openposeDetection.joints[jb]:
                    each_paf.append(numbers.pop(-1))
                paf.append(each_paf)
            openposeDetection.pafs.append(paf)
        openposeDetections.append(openposeDetection)
    assert len(numbers)==0
    return openposeDetections

def read_openpose_detections(txt_paths):
    # 从 txt 中读取
    openpose_detection_lists=[
        read_openpose_detection(txt_path)
        for txt_path in txt_paths
    ]
    # map SKEL25 to SKEL19
    for openpose_detections in openpose_detection_lists:
        for openpose_detection in openpose_detections:
            openpose_detection.mapping()
    # convert x,y from ratio to pixel coord
    resolution=(2048,2048) # width,height
    for openpose_detections in openpose_detection_lists:
        for openpose_detection in openpose_detections:
            for joints in openpose_detection.joints:
                for joint in joints:
                    joint[0],joint[1]=joint[0]*resolution[0],joint[1]*resolution[1]
    return list(zip(*openpose_detection_lists))

    
def main():
    txt_path="./data/seq_3/detection/18181923.txt"
    openposeDetections=read_openpose_detection(txt_path)
    for openposeDetection in openposeDetections:
        openposeDetection.mapping()
    import pdb;pdb.set_trace()

if __name__=="__main__":
    main()