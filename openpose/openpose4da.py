import os
import sys
import math

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
from torch import nn
from joblib import Parallel,delayed

from .util import padRightDownCorner
from .model import bodypose_25_model
from association.openpose_detection import SKEL25DEF

class torch_openpose(nn.Module):
    def __init__(self,model_body25 = './weight/body_25.pth',try_cuda=True,lock=None):
        super().__init__()
        self.try_cuda=try_cuda
        self.lock=lock

        self.model = bodypose_25_model()
        self.model.load_state_dict(torch.load(model_body25))
        if self.try_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        
        self.limbSeq = SKEL25DEF.paf_dict
        self.mapIdx = [
            [0,1], 
            [2,3],
            [4,5],
            [6,7],
            [8,9],
            [10,11],
            [12,13],
            [14,15],
            [16,17],
            [18,19],
            [20,21],
            [22,23],
            [24,25],
            [26,27],
            [28,29],
            [30,31],
            [32,33],
            [34,35],
            [36,37],
            [38,39],
            [40,41],
            [42,43],
            [44,45],
            [46,47],
            [48,49],
            [50,51]
        ]
        self.njoint = SKEL25DEF.joint_size + 1 # 最后一个是背景
        self.npaf = SKEL25DEF.paf_size*2

    def get_heatmap_and_paf(self,m,multiplier,oriImg,stride,padValue):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
        im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        if self.lock is not None:
            self.lock.acquire()
        data = torch.from_numpy(im).float()
        if self.try_cuda and torch.cuda.is_available():
            data = data.cuda()
        # data = data.permute([2, 0, 1]).unsqueeze(0).float()
        with torch.no_grad():
            heatmap, paf = self.model(data)
        
        heatmap = heatmap.detach().cpu().numpy()
        paf = paf.detach().cpu().numpy()
        if self.lock is not None:
            self.lock.release()

        # extract outputs, resize, and remove padding
        # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
        paf = np.transpose(np.squeeze(paf), (1, 2, 0))  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        return heatmap,paf
    
    def get_peak(self,heatmap_avg,part,thre1):
        map_ori = heatmap_avg[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [list(x)+[map_ori[x[1], x[0]]] for x in peaks]
        return peaks_with_score
    
    def get_connection_candidate(self,paf_avg,k,all_peaks,mid_num):
        score_mid = paf_avg[:, :, self.mapIdx[k]]
        candA = all_peaks[self.limbSeq[k][0]]
        candB = all_peaks[self.limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        connection_candidate = []
        for i in range(nA):
            connections=[]
            for j in range(nB):
                vec = np.subtract(candB[j][:2], candA[i][:2])
                norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-10 # 防止为0
                vec = np.divide(vec, norm)

                startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])

                score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1]) # 两个单位向量的点乘
                score_paf=score_midpts.mean()
                score_paf=min(max(score_paf,0.0),1.0) # 限定 paf score 的范围为 0~1
                connections.append(score_paf)
            connection_candidate.append(connections)
        return connection_candidate

    def forward(self, oriImg,scale_search = [0.7, 1.0, 1.3]): # # issues this scale search option is better
        boxsize = 368 # base resolution
        stride = 8
        padValue = 128
        thre1 = 0.1

        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]

        maps=Parallel(n_jobs=1,backend="threading")( # len(multiplier)
            delayed(self.get_heatmap_and_paf)(m,multiplier,oriImg,stride,padValue)
            for m in range(len(multiplier))
        )
        heatmaps,pafs=list(zip(*maps))
        heatmaps=np.stack(heatmaps)
        pafs=np.stack(pafs)
        heatmap_avg=np.mean(heatmaps,axis=0)
        paf_avg=np.mean(pafs,axis=0)

        # heatmap_avg paf_avg 是融合了多尺度的 heatmap 与 paf
        all_peaks=Parallel(n_jobs=self.njoint-1,backend="threading")(
            delayed(self.get_peak)(heatmap_avg,part,thre1)
            for part in range(self.njoint - 1)
        )

        mid_num = 10 # 积分的采样数

        connection_all=Parallel(n_jobs=len(self.mapIdx),backend="threading")(
            delayed(self.get_connection_candidate)(paf_avg,k,all_peaks,mid_num)
            for k in range(len(self.mapIdx))
        )
        
        return all_peaks,connection_all


