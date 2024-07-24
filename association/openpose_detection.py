import os
import sys
from typing import Union
import copy

from easydict import EasyDict

# openpose/src/openpose/pose/poseParameters.cpp

SKEL19DEF=EasyDict({ # use 4da 19 keypoints definition 
    'joint_size':19,
    'paf_size': 18,
    'joint_names': [
        'midhip', 'neck', 'r_hip', 'l_hip', 'nose', 
        'r_shoulder', 'l_shoulder', 'r_knee', 'l_knee', 'r_ear', 
        'l_ear', 'r_elbow', 'l_elbow', 'r_ankle', 'l_ankle', 
        'r_wrist', 'l_wrist', 'l_bigtoe', 'r_bigtoe'
    ],
    'paf_dict':[
        [1,0],
        [2,7],
        [7,13],
        [0,2],
        [0,3],
        [3,8],
        [8,14],
        [1,5],
        [5,11],
        [11,15],
        [5,9],
        [1,6],
        [6,12],
        [12,16],
        [6,10],
        [1,4],
        [14,17],
        [13,18],
    ]
})

SKEL25DEF=EasyDict({
    'joint_size':25,
    'paf_size':26,
    'joint_names':[
        'nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 
        'l_shoulder', 'l_elbow', 'l_wrist', 'midhip', 'r_hip', 
        'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 
        'r_eye', 'l_eye', 'r_ear', 'l_ear', 'l_bigtoe', 
        'l_smalltoe', 'l_heel', 'r_bigtoe', 'r_smalltoe', 'r_heel'
    ],
    'paf_dict':[
        [1, 8],
        [9, 10],
        [10, 11],
        [8, 9],
        [8, 12],
        [12, 13],
        [13, 14],
        [1, 2],
        [2, 3],
        [3, 4],
        [2, 17],
        [1, 5],
        [5, 6],
        [6, 7],
        [5, 18],
        [1, 0],
        [0, 15],
        [0, 16],
        [15, 17],
        [16, 18],
        [14, 19],
        [19, 20],
        [14, 21],
        [11, 22],
        [22, 23],
        [11, 24]
    ]
})

class OpenposeDetection(object):
    def __init__(self,definition=None) -> None:
        self.joints=None
        self.pafs=None
        self.definition=definition
    
    def mapping(self,):
        # from BODY25 to SKEL19
        jointMapping=[4, 1, 5, 11, 15, 6, 12, 16, 0, 2, 7, 13, 3, 8, 14, -1, -1, 9, 10, 17, -1, -1, 18, -1, -1]
        pafMapping=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, 16, -1, -1, 17, -1, -1]
        joints=[None for _ in range(SKEL19DEF.joint_size)]
        pafs=[None for _ in range(SKEL19DEF.paf_size)]
        for body25_index,skel19_index in enumerate(jointMapping):
            if skel19_index==-1:
                continue
            joints[skel19_index]=self.joints[body25_index]
        for body25_index,skel19_index in enumerate(pafMapping):
            if skel19_index==-1:
                continue
            pafs[skel19_index]=self.pafs[body25_index]
        self.joints=joints
        self.pafs=pafs
