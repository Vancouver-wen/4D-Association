import os
import sys

import numpy as np
import cv2
from loguru import logger

from .camera import line2linedist, point2linedist,Camera
from .openpose_detection import SKEL19DEF,SKEL25DEF

class GraphConstruct():

    def __init__(
        self,
        cameras:list[Camera],
        max_epi_dist: float = 0.15,
        max_temp_dist: float = 0.2,
        normalize_edges: bool = True,
        definition=SKEL19DEF,
    ) -> None:
        """
        Args:
            cameras list[Camera]:
                camera param for undistorting and ray casting
            n_kps (int):
                keypoints number
            n_pafs (int):
                paf number
            max_epi_dist (float):
                maximal epipolar distance to be accepted
            max_temp_dist (float):
                maximal temporal tracking distance to be accepted
            normalize_edges (bool):
                indicator to normalize all edges
        """
        self.definition=definition
        self.n_views = len(cameras)
        self.n_kps = self.definition.joint_size
        self.n_pafs = self.definition.paf_size
        self.max_epi_dist = max_epi_dist
        self.max_temp_dist = max_temp_dist
        self.normalize_edges = normalize_edges
        self.paf_dict = self.definition.paf_dict
        self.m_epi_edges = {
            i: {
                j: {k: -1
                    for k in range(self.n_views)}
                for j in range(self.n_views)
            }
            for i in range(self.n_kps)
        }
        self.m_temp_edges = {
            i: {j: -1
                for j in range(self.n_views)}
            for i in range(self.n_kps)
        }
        self.m_kps_rays = {
            i: {j: []
                for j in range(self.n_kps)}
            for i in range(self.n_views)
        }
        self.m_bone_nodes = {
            i: {j: []
                for j in range(self.n_views)}
            for i in range(self.n_pafs)
        }
        self.m_bone_epi_edges = {
            i: {
                j: {k: []
                    for k in range(self.n_views)}
                for j in range(self.n_views)
            }
            for i in range(self.n_pafs)
        }
        self.m_bone_temp_edges = {
            i: {j: []
                for j in range(self.n_views)}
            for i in range(self.n_pafs)
        }

        self.cameras = cameras
        self.last_multi_kps3d = dict()

    def __call__(self, kps2d, pafs, last_multi_kps3d=dict()):
        """construct the 4D graph.
        Args:
            kps2d (list): 2D keypoints -> [view][joint_type][joint_index] -> [x,y,score]:list[float,float,float]
            pafs (list): part affine field -> [view][paf_type][joint_a_index][joint_b_index] -> paf_score:float
            last_multi_kps3d (dict): 3D keypoints of last frame
        Returns:
            graph (dict): the constructed 4D graph
        """
        self.kps2d = kps2d
        self.pafs = pafs
        self.last_multi_kps3d = last_multi_kps3d
        self.construct_graph()
        output=dict(
            m_epi_edges=self.m_epi_edges,
            m_temp_edges=self.m_temp_edges,
            m_bone_nodes=self.m_bone_nodes,
            m_bone_epi_edges=self.m_bone_epi_edges,
            m_bone_temp_edges=self.m_bone_temp_edges
        )
        return output

    def construct_graph(self):
        self._calculate_kps_rays()
        self._calculate_paf_edges()
        self._calculate_epi_edges()
        self._calculate_temp_edges()

        self._calculate_bone_nodes()
        self._calculate_bone_epi_edges()
        self._calculate_bone_temp_edges()

    def _calculate_kps_rays(self):
        for view in range(self.n_views):
            cam = self.cameras[view]
            for kps_id in range(self.n_kps):
                self.m_kps_rays[view][kps_id] = []
                kps = self.kps2d[view][kps_id]
                for kps_candidate in range(len(kps)):
                    self.m_kps_rays[view][kps_id].append(cam.cal_ray(kps[kps_candidate][:2]))

    def _calculate_paf_edges(self):
        if self.normalize_edges:
            for paf_id in range(self.n_pafs):
                for detection in self.pafs:
                    pafs = detection[paf_id]
                    if not isinstance(pafs,(np.ndarray, np.generic)):
                        pafs=np.array(pafs,dtype=np.float32)
                    if np.sum(pafs) > 0:
                        row_factor = np.clip(pafs.sum(1), 1.0, None)
                        col_factor = np.clip(pafs.sum(0), 1.0, None)
                        for i in range(len(row_factor)):
                            pafs[i] /= row_factor[i]
                        for j in range(len(col_factor)):
                            pafs[:, j] /= col_factor[j]
                    detection[paf_id] = pafs

    def _calculate_epi_edges(self):
        for kps_id in range(self.n_kps):
            for view1 in range(self.n_views - 1):
                cam1 = self.cameras[view1]
                for view2 in range(view1 + 1, self.n_views):
                    cam2 = self.cameras[view2]
                    kps1 = self.kps2d[view1][kps_id]
                    kps2 = self.kps2d[view2][kps_id]
                    ray1 = self.m_kps_rays[view1][kps_id]
                    ray2 = self.m_kps_rays[view2][kps_id]

                    if len(kps1) > 0 and len(kps2) > 0:
                        epi = np.full((len(kps1), len(kps2)), -1.0)
                        for kps1_candidate in range(len(kps1)):
                            for kps2_candidate in range(len(kps2)):
                                dist = line2linedist(cam1.Pos,ray1[kps1_candidate],cam2.Pos,ray2[kps2_candidate])
                                if dist < self.max_epi_dist:
                                    epi[kps1_candidate,kps2_candidate] = 1 - dist / self.max_epi_dist

                        if self.normalize_edges:
                            row_factor = np.clip(epi.sum(1), 1.0, None)
                            col_factor = np.clip(epi.sum(0), 1.0, None)
                            for i in range(len(row_factor)):
                                epi[i] /= row_factor[i]
                            for j in range(len(col_factor)):
                                epi[:, j] /= col_factor[j]
                        self.m_epi_edges[kps_id][view1][view2] = epi
                        self.m_epi_edges[kps_id][view2][view1] = epi.T

    def _calculate_temp_edges(self):
        for kps_id in range(self.n_kps):
            for view in range(self.n_views):
                rays = self.m_kps_rays[view][kps_id]
                if len(self.last_multi_kps3d) > 0 and len(rays) > 0:
                    temp = np.full((len(self.last_multi_kps3d), len(rays)),-1.0)
                    for pid, person_id in enumerate(self.last_multi_kps3d): # pid 代表了 person_id 在 self.last_multi_kps3d 中的 keys() 顺序
                        limb = self.last_multi_kps3d[person_id]
                        if limb[kps_id][3] > 0:
                            for kps_candidate in range(len(rays)):
                                dist = point2linedist(limb[kps_id][:3],self.cameras[view].Pos,rays[kps_candidate])
                                if dist < self.max_temp_dist:
                                    temp[pid,kps_candidate] = 1 - dist / self.max_temp_dist
                    if self.normalize_edges:
                        row_factor = np.clip(temp.sum(1), 1.0, None)
                        col_factor = np.clip(temp.sum(0), 1.0, None)
                        for i in range(len(row_factor)):
                            temp[i] /= row_factor[i]
                        for j in range(len(col_factor)):
                            temp[:, j] /= col_factor[j]
                    self.m_temp_edges[kps_id][view] = temp

    def _calculate_bone_nodes(self):
        for paf_id in range(self.n_pafs):
            kps1, kps2 = self.paf_dict[paf_id]
            for view in range(self.n_views):
                self.m_bone_nodes[paf_id][view] = []
                for kps1_candidate in range(len(self.kps2d[view][kps1])):
                    for kps2_candidate in range(len(self.kps2d[view][kps2])):
                        if self.pafs[view][paf_id][kps1_candidate][kps2_candidate] > 0:
                            self.m_bone_nodes[paf_id][view].append((kps1_candidate, kps2_candidate))

    def _calculate_bone_epi_edges(self):
        for paf_id in range(self.n_pafs):
            kps_pair = self.paf_dict[paf_id]
            for view1 in range(self.n_views - 1):
                for view2 in range(view1 + 1, self.n_views):
                    nodes1 = self.m_bone_nodes[paf_id][view1]
                    nodes2 = self.m_bone_nodes[paf_id][view2]
                    epi = np.full((len(nodes1), len(nodes2)), -1.0)
                    for bone1_id in range(len(nodes1)):
                        for bone2_id in range(len(nodes2)):
                            node1 = nodes1[bone1_id]
                            node2 = nodes2[bone2_id]
                            epidist = np.zeros(2)
                            for i in range(2):
                                epidist[i] = self.m_epi_edges[kps_pair[i]][view1][view2][node1[i],node2[i]]
                            if epidist.min() < 0:
                                continue
                            epi[bone1_id, bone2_id] = epidist.mean()
                    self.m_bone_epi_edges[paf_id][view1][view2] = epi
                    self.m_bone_epi_edges[paf_id][view2][view1] = epi.T

    def _calculate_bone_temp_edges(self):
        for paf_id in range(self.n_pafs):
            kps_pair = self.paf_dict[paf_id]
            for view in range(self.n_views):
                nodes = self.m_bone_nodes[paf_id][view]
                temp = np.full((len(self.last_multi_kps3d), len(nodes)), -1.0)
                for pid in range(len(temp)):
                    for node_candidate in range(len(nodes)):
                        node = nodes[node_candidate]
                        tempdist = []
                        for i in range(2):
                            tempdist.append(self.m_temp_edges[kps_pair[i]][view][pid][node[i]]) # pid 代表了 person_id 在 self.last_multi_kps3d 中的 keys() 顺序
                        if min(tempdist) > 0:
                            temp[pid,node_candidate] = sum(tempdist) / len(tempdist)
                self.m_bone_temp_edges[paf_id][view] = temp
