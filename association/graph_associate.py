import copy
import heapq
import math
import json
import time

import numpy as np
from joblib import Parallel,delayed
from natsort import natsorted

from .camera import line2linedist, point2linedist,Camera
from .openpose_detection import SKEL19DEF,SKEL25DEF

def welsch(c, x):
    x = x / c
    return 1 - math.exp(-x * x / 2)

class Clique():

    def __init__(self, paf_id:int, proposal:list[int], score:float=-1) -> None:
        """class for limb clique, which is used for solve 4D graph.

        Args:
            paf_id (int): the paf index
            paf index proposal (List): a list of allocated bone index to the clique
            score (float): the score of the clique, larger score will be solved earlier
        """
        self.paf_id = paf_id
        self.proposal = proposal
        self.score = score

    def __lt__(self, other):
        if self.score > other.score:
            return True
        else:
            return False
        
    def __str__(self) -> str:
        return str({
            'paf_id':self.paf_id,
            'proposal':self.proposal,
            'score':self.score
        })


class Voting():

    def __init__(self) -> None:
        """
        vote class for clique 
        it will record the kps haven been allocated and it will be used to solve graph.
        """
        # 这里使用int8会导致数组越界 -> use int32
        self.fst = np.zeros(2, dtype=np.int32) # first
        self.sec = np.zeros(2, dtype=np.int32) # second
        self.fst_cnt = np.zeros(2, dtype=np.int32) # first count
        self.sec_cnt = np.zeros(2, dtype=np.int32) # second count
        self.vote = dict()

    def parse(self):
        """
        统计self.vote中前两位的投票数目
        """
        self.fst_cnt = np.zeros(2)
        self.sec_cnt = np.zeros(2)
        if len(self.vote) == 0:
            return
        _vote = copy.deepcopy(self.vote)
        for i in range(2):
            for index in range(2):
                person_id = max(_vote, key=lambda x: _vote[x][index])

                if i == 0:
                    self.fst[index] = person_id
                    self.fst_cnt[index] = _vote[person_id][index]
                else:
                    self.sec[index] = person_id
                    self.sec_cnt[index] = _vote[person_id][index]
                _vote[person_id][index] = 0
    
    def __str__(self) -> str:
        return str({
            'self.fst':self.fst,
            'self.sec':self.sec,
            'self.fst_cnt':self.fst_cnt,
            'self.sec_cnt':self.sec_cnt,
            'self.vote':self.vote
        })


class GraphAssociate():

    def __init__(self,
        n_views:int,
        w_epi: float = 2,
        w_temp: float = 2,
        w_view: float = 2,
        w_paf: float = 1,
        w_hier: float = 0.5,
        c_view_cnt: float = 1.5,
        min_check_cnt: int = 1,
        definition=SKEL19DEF,
    ) -> None:
        """
        Args:
            n_views (int):
                views number of dataset
            n_kps (int):
                keypoints number
            n_pafs (int):
                paf number
            w_epi (float):
                clique score weight for epipolar distance
            w_temp (float):
                clique score weight for temporal tracking distance
            w_view (float):
                clique score weight for view number
            w_paf (float):
                clique score weight for paf edge
            w_hier (float):
                clique score weight for hierarchy
            c_view_cnt (float):
                maximal view number
            min_check_cnt (int):
                minimum check number
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.definition=definition
        self.n_views = n_views
        self.n_kps = self.definition.joint_size
        self.n_pafs = self.definition.paf_size
        self.w_epi = w_epi
        self.w_temp = w_temp
        self.w_view = w_view
        self.w_paf = w_paf
        self.c_view_cnt = c_view_cnt
        self.min_check_cnt = min_check_cnt
        self.paf_dict = self.definition.paf_dict

        self.m_kps2paf = {i: [] for i in range(self.n_kps)}
        for paf_id in range(self.n_pafs):
            kps_pair = self.paf_dict[paf_id]
            self.m_kps2paf[kps_pair[0]].append(paf_id)
            self.m_kps2paf[kps_pair[1]].append(paf_id)

        self.m_assign_map = { # m_assign_map 记录每个 kpts被分配到哪一个 person_id。 使用-1表示没有被分配 person_id
            i: {j: []
                for j in range(self.n_kps)}
            for i in range(self.n_views)
        }
        self.mpersons_map = dict()

        self.last_multi_kps3d = dict()
        self.cliques = []

    def __call__(self, kps2d, pafs, graph, last_multi_kps3d=dict):
        """associate keypoint in multiply view.
        Args:
            kps2d (list): 2D keypoints
            pafs (list): part affine field
            graph (list): the 4D graph to be associated
            last_multi_kps3d (dict): 3D keypoints of last frame
        Returns:
            mpersons_map (dict): the associate limb
        """
        self.kps2d = kps2d
        self.pafs = pafs

        self.m_epi_edges = graph['m_epi_edges']
        self.m_temp_edges = graph['m_temp_edges']
        self.m_bone_nodes = graph['m_bone_nodes']
        self.m_bone_epi_edges = graph['m_bone_epi_edges']
        self.m_bone_temp_edges = graph['m_bone_temp_edges']

        self.last_multi_kps3d = last_multi_kps3d
        self.solve_graph()

        return self.mpersons_map

    def solve_graph(self):
        self.initialize()
        start=time.time()
        self.enumerate_cliques()
        print(f"enumerate cliques time consume: {time.time()-start}")
        start=time.time()
        number=0
        while len(self.cliques) > 0:
            number+=1
            self.assign_top_clique()
        print(f"{number} loop assign top clique time consume: {time.time()-start}")

    def initialize(self):
        for kps_id in range(self.n_kps):
            for view in range(self.n_views):
                self.m_assign_map[view][kps_id] = np.full(len(self.kps2d[view][kps_id]), -1)

        self.mpersons_map = {}
        for person_id in self.last_multi_kps3d:
            self.mpersons_map[person_id] = np.full((self.n_kps, self.n_views),-1)

    def enumerate_paf_cliques(self,paf_id):
        cliques=[]
        nodes = self.m_bone_nodes[paf_id]
        pick = [-1] * (self.n_views + 1)
        available_node = {
            i: {j: []
                for j in range(self.n_views + 1)}
            for i in range(self.n_views + 1)
        }
        index = -1
        while True:
            if index >= 0 and pick[index] >= len(available_node[index][index]):
                pick[index] = -1
                index = index - 1
                if index < 0:
                    break
                pick[index] += 1
            elif index == len(pick) - 1:
                if sum(pick[:self.n_views]) != -self.n_views:
                    clique = Clique(
                        paf_id=paf_id, 
                        proposal=[-1] * len(pick),
                        score=-1.0
                    )
                    for i in range(len(pick)):
                        if pick[i] != -1:
                            if i == len(pick) - 1:
                                clique.proposal[i] = list(self.last_multi_kps3d.keys())[available_node[i][i][pick[i]]] # pid -> person_id
                            else:
                                clique.proposal[i] = available_node[i][i][pick[i]]
                    clique.score = self.cal_clique_score(clique)
                    cliques.append(clique)
                pick[index] += 1
            else:
                index += 1
                # update available nodes
                if index == 0:
                    for view in range(self.n_views):
                        for bone in range(len(nodes[view])):
                            available_node[0][view].append(bone)
                    for pid in range(len(self.last_multi_kps3d)):
                        available_node[0][self.n_views].append(pid)
                else:
                    # epipolar constrain
                    if pick[index - 1] >= 0:
                        for view in range(index, self.n_views):
                            available_node[index][view] = []
                            epiEdges = self.m_bone_epi_edges[paf_id][index - 1][view]
                            bone1_id = available_node[index -1][index -1][pick[index -1]]
                            for bone2_id in available_node[index -1][view]:
                                if epiEdges[bone1_id, bone2_id] > 0:
                                    available_node[index][view].append(bone2_id)
                    else:
                        for view in range(index, self.n_views):
                            available_node[index][view] = available_node[index - 1][view][:]
                    # temporal constrain
                    if pick[self.n_views - 1] > 0:
                        available_node[index][self.n_views] = []
                        temp_edge = self.m_bone_temp_edges[paf_id][self.n_views - 1]
                        bone1_id = available_node[self.n_views -1][self.n_views -1][pick[self.n_views -1]]
                        for pid in available_node[index - 1][self.n_views]:
                            if temp_edge[pid, bone1_id] > 0:
                                available_node[index][self.n_views].append(pid)
                    else:
                        available_node[index][self.n_views] = available_node[index - 1][self.n_views][:]
        cliques=list(filter(lambda x:(np.array(x.proposal)!=-1).sum()>=2,cliques))
        cliques=natsorted(cliques,key=lambda x:x.score,reverse=True)[:min(100,len(cliques))]
        # for clique in cliques[:20]:
        #     print(clique)
        # import pdb;pdb.set_trace()
        return cliques

    def enumerate_cliques(self):
        tmp_cliques=Parallel(n_jobs=self.n_pafs)( # n_jobs=self.n_pafs
            delayed(self.enumerate_paf_cliques)(paf_id)
            for paf_id in range(self.n_pafs)
        )
        for paf_id in range(self.n_pafs):
            self.cliques.extend(tmp_cliques[paf_id])
        heapq.heapify(self.cliques)

    def allocFlag(self,clique,nodes,kps_pair):
        if sum(np.array(clique.proposal) >= 0) == 0:
            return True
        view_var = max(clique.proposal)
        view = clique.proposal.index(view_var)
        node = nodes[view][clique.proposal[view]]
        person_candidate = []
        for person_id in self.mpersons_map:
            def check_cnt():
                cnt = 0
                for i in range(2):
                    _cnt = self.check_kpt_compatibility(view, kps_pair[i], node[i], person_id)
                    if _cnt == -1:
                        return -1
                    cnt += _cnt
                return cnt
            cntt = check_cnt()
            if cntt >= self.min_check_cnt:
                person_candidate.append([cntt, person_id])
        if len(person_candidate) == 0:
            return True
        person_id = max(person_candidate)[1]
        person = self.mpersons_map[person_id]
        for i in range(2):
            person[kps_pair[i], view] = node[i]
            self.m_assign_map[view][kps_pair[i]][node[i]] = person_id

        self.mpersons_map[person_id] = person
        return False

    def assign_top_clique(self):
        """
        clique:
            {'paf_id': 6, 'proposal': [0, 1, 1, 0, -1, 2, -1], 'score': 0.6756603975043303}
        nodes:
            {0: [(0, 0), (1, 1), (2, 2)], 1: [(0, 1), (1, 2), (2, 3)], 2: [(1, 1), (2, 2)], 3: [(0, 1), (1, 2), (1, 3)], 4: [(0, 0), (1, 1)], 5: [(0, 1), (0, 2), (1, 0), (2, 2)]}
        kps_pair:
            [8, 14]
        """
        clique = heapq.heappop(self.cliques)
        nodes = self.m_bone_nodes[clique.paf_id]
        kps_pair = self.paf_dict[clique.paf_id]
        if clique.proposal[self.n_views] != -1: 
            # 在时序关系中找到了对应的骨骼(这个骨骼带有person_id信息)
            person_id = clique.proposal[self.n_views]
            if self.check_cnt(clique, kps_pair, nodes, person_id) != -1: # 检查兼容性
                person = self.mpersons_map[person_id]
                # 1. 如果self.mersons_map[person_id] 全是 -1 , 则 check_cnt 一定能通过
                # 2. 否则 就会检查与现有的 self.mpersons_map[person_id] 的兼容性
                # 3. 如果兼容性检查通过, 则 更新合并关键点
                _proposal = [-1] * (self.n_views + 1)
                for view in range(self.n_views):
                    if clique.proposal[view] != -1:
                        node = nodes[view][clique.proposal[view]]
                        assign = ( self.m_assign_map[view][kps_pair[0]][node[0]] , self.m_assign_map[view][kps_pair[1]][node[1]] ) 
                        if (assign[0] == -1 or assign[0] == person_id) and (assign[1] == -1 or assign[1] == person_id): # 两个端点都没有被分配 -> 合并
                            for i in range(2):
                                person[kps_pair[i], view] = node[i]
                                self.m_assign_map[view][kps_pair[i]][node[i]] = person_id
                        else:
                            _proposal[view] = clique.proposal[view] # 任一个端点已经被分配了 -> 该limb需要重新塞到 heap 中
                self.mpersons_map[person_id] = person
                self.push_clique(clique.paf_id, _proposal[:])
            else:
                _proposal = clique.proposal
                _proposal[self.n_views] = -1 # 不兼容 -> 剥夺 时序骨骼信息
                self.push_clique(clique.paf_id, _proposal[:])
        else:
            # 在时序关系中没有找到对应的骨骼
            voting = Voting()
            voting = self.clique2voting(clique, voting)
            voting.parse()
            # import pdb;pdb.set_trace()
            # 一个 limb_clique 应该包含两个 keypoint_clique: A & B
            if sum(voting.fst_cnt) == 0:
                # ('1. A & B not assigned yet')
                # limb_clique中 两个keypoint_clique都没有被分配 person_id -> 创建新的person_id
                if self.allocFlag(clique,nodes,kps_pair):
                    person = np.full((self.n_kps, self.n_views), -1)
                    person_id=0 if len(self.mpersons_map) == 0 else max(self.mpersons_map) + 1
                    for view in range(self.n_views):
                        if clique.proposal[view] >= 0:
                            node = nodes[view][clique.proposal[view]]
                            for i in range(2):
                                person[kps_pair[i], view] = node[i]
                                self.m_assign_map[view][kps_pair[i]][node[i]] = person_id
                    self.mpersons_map[person_id] = person
            elif min(voting.fst_cnt) == 0:
                # ('2. A assigned but not B: Add B to person with A ')
                valid_id = 0 if voting.fst_cnt[0] > 0 else 1
                master_id = voting.fst[valid_id]
                unassignj_id = kps_pair[1 - valid_id]
                person = self.mpersons_map[master_id]
                _proposal = [-1] * (self.n_views + 1)
                for view in range(self.n_views):
                    if clique.proposal[view] >= 0:
                        node = nodes[view][clique.proposal[view]]
                        unassignj_candidata = node[1 - valid_id]
                        assigned = self.m_assign_map[view][kps_pair[valid_id]][node[valid_id]]
                        if assigned == master_id: # 已经被分配的person_id的点 与 该点所在clique投票得到的person_id 是一致
                            if person[unassignj_id, view] == -1 and self.check_kpt_compatibility(view, unassignj_id,unassignj_candidata, master_id) >= 0:
                                person[unassignj_id,view] = unassignj_candidata
                                self.m_assign_map[view][unassignj_id][unassignj_candidata] = master_id
                            else:
                                continue
                        elif assigned == -1 and voting.fst_cnt[valid_id] >= 2 and voting.sec_cnt[valid_id] == 0\
                                and (person[kps_pair[0], view] == -1 or person[kps_pair[0], view] == node[0])\
                                and (person[kps_pair[1], view] == -1 or person[kps_pair[1], view] == node[1]):
                            # keypoint还没有被分配person_id 并且 clique投票了一个person_id
                            if self.check_kpt_compatibility(view, kps_pair[0], node[0], master_id) >= 0 and self.check_kpt_compatibility(view, kps_pair[1], node[1],master_id) >= 0:
                                for i in range(2):
                                    person[kps_pair[i], view] = node[i]
                                    self.m_assign_map[view][kps_pair[i]][node[i]] = master_id
                            else:
                                _proposal[view] = clique.proposal[view]
                        else:
                            _proposal[view] = clique.proposal[view]
                self.mpersons_map[master_id] = person
                if _proposal != clique.proposal:
                    self.push_clique(clique.paf_id, _proposal[:])
            elif voting.fst[0] == voting.fst[1]:
                # ('4. A & B already assigned to same person')
                master_id = voting.fst[0]
                person = self.mpersons_map[master_id]
                _proposal = [-1] * (self.n_views + 1)
                for view in range(self.n_views):
                    if clique.proposal[view] >= 0:
                        node = nodes[view][clique.proposal[view]]
                        assign_id = [
                            self.m_assign_map[view][kps_pair[0]][node[0]],
                            self.m_assign_map[view][kps_pair[1]][node[1]]
                        ]
                        if assign_id[0] == master_id and assign_id[1] == master_id:
                            continue
                        elif self.check_kpt_compatibility(view, kps_pair[0], node[0], master_id) == -1 or self.check_kpt_compatibility(view, kps_pair[1], node[1], master_id) == -1:
                            _proposal[view] = clique.proposal[view]
                        elif (assign_id[0] == master_id and assign_id[1]== -1) or (assign_id[0] == -1 and assign_id[1] == master_id):
                            valid_id = 0 if assign_id[1] == -1 else 1
                            unassignj_id = kps_pair[1 - valid_id]
                            unassignj_candidata = node[1 - valid_id]
                            if person[unassignj_id, view] == -1 or person[unassignj_id, view] == unassignj_candidata:
                                person[unassignj_id,view] = unassignj_candidata
                                self.m_assign_map[view][unassignj_id][unassignj_candidata] = master_id
                            else:
                                _proposal[view] = clique.proposal[view]
                        elif max(assign_id) == -1 and sum(voting.sec_cnt) == 0 and (person[kps_pair[0], view] == -1 or person[kps_pair[0], view] == node[0]) and (person[kps_pair[1], view] == -1 or person[kps_pair[1], view] == node[1]):
                            for i in range(2):
                                person[kps_pair[i], view] = node[i]
                                self.m_assign_map[view][kps_pair[i]][node[i]] = master_id
                        else:
                            _proposal[view] = clique.proposal[view]
                    if _proposal != clique.proposal:
                        self.push_clique(clique.paf_id, _proposal[:])
                self.mpersons_map[master_id] = person
            else:
                # ('5. A & B already assigned to different people')
                for index in range(2):
                    while voting.sec_cnt[index] != 0:
                        master_id = min(voting.fst[index], voting.sec[index])
                        slave_id = max(voting.fst[index], voting.sec[index])
                        assert slave_id <= max(self.mpersons_map)
                        if self.check_person_compatibility(master_id, slave_id) >= 0:
                            self.merge_person(master_id, slave_id)
                            voting = self.clique2voting(clique, voting)
                            voting.parse()
                        else:
                            voting.vote[
                                voting.fst[index]][index] = voting.vote[voting.sec[index]][index] = 0
                            iter = max(voting.vote,key=lambda x: voting.vote[x][index])
                            voting.sec[index] = iter
                            voting.sec_cnt[index] = voting.vote[iter][index]
                if voting.fst[0] != voting.fst[1]:
                    conflict = [0] * self.n_views
                    master_id = min(voting.fst)
                    slave_id = max(voting.fst)
                    for view in range(self.n_views):
                        conflict[view] = 1 if self.check_person_compatibility_sview(master_id, slave_id, view) == -1 else 0
                    if sum(conflict) == 0:
                        self.merge_person(master_id, slave_id)
                        _proposal = [-1] * (self.n_views + 1)
                        master = self.mpersons_map[master_id]
                        for view in range(self.n_views):
                            if clique.proposal[view] >= 0:
                                assert clique.proposal[view] < len(nodes[view])
                                node = nodes[view][clique.proposal[view]]
                                if master[kps_pair[0],view] != node[0] or master[kps_pair[1], view] != node[1]:
                                    _proposal[view] = clique.proposal[view]
                        self.push_clique(clique.paf_id, _proposal[:])
                    else:
                        _proposal_pair = np.full((self.n_views + 1, 2), -1)
                        for i in range(len(conflict)):
                            _proposal_pair[i, conflict[i]] = clique.proposal[i]
                        if min(_proposal_pair[:, 0]) >= 0 and min(_proposal_pair[:, 1]) >= 0:
                            self.push_clique(clique.paf_id,_proposal_pair[:, 0].copy())
                            self.push_clique(clique.paf_id,_proposal_pair[:, 1].copy())
                        elif sum(np.array(clique.proposal[:self.n_views]) >= 0) > 1:
                            for i in range(len(conflict)):
                                _proposal = [-1] * (self.n_views + 1)
                                _proposal[i] = clique.proposal[i]
                                self.push_clique(clique.paf_id, _proposal[:])

    def cal_clique_score(self, clique):
        # epipolar score
        scores = []
        for view1 in range(self.n_views - 1):
            if clique.proposal[view1] == -1:
                continue
            for view2 in range(view1 + 1, self.n_views):
                if clique.proposal[view2] == -1:
                    continue
                scores.append(self.m_bone_epi_edges[clique.paf_id][view1][view2][clique.proposal[view1],clique.proposal[view2]])

        if len(scores) > 0:
            epi_score = sum(scores) / len(scores)
        else:
            epi_score = 1
        # temporal score
        scores = []
        person_id = clique.proposal[self.n_views]
        if person_id != -1:
            for view in range(self.n_views):
                if clique.proposal[view] == -1:
                    continue
                scores.append(self.m_bone_temp_edges[clique.paf_id][view][list(self.last_multi_kps3d.keys()).index(person_id),clique.proposal[view]]) # person_id -> pid
        if len(scores) > 0:
            temp_score = sum(scores) / len(scores)
        else:
            temp_score = 0
        # paf score
        scores = []
        for view in range(self.n_views):
            if clique.proposal[view] == -1:
                continue
            candidata_bone = self.m_bone_nodes[clique.paf_id][view][clique.proposal[view]]
            scores.append(self.pafs[view][clique.paf_id][candidata_bone[0]][candidata_bone[1]])
        paf_score = sum(scores) / len(scores)
        # view score
        var = sum(np.array(clique.proposal[:self.n_views]) >= 0)
        view_score = welsch(self.c_view_cnt, var)
        # hier score ignored
        score=(self.w_epi * epi_score + self.w_temp * temp_score + self.w_paf * paf_score + self.w_view * view_score) / (self.w_epi + self.w_temp + self.w_paf + self.w_view)
        return score

    def check_cnt(self, clique, kps_pair, nodes, person_id):
        """
        时序骨骼的两个端点 与 该clique 不允许发生一个冲突
        如果有冲突,就应剥夺clique中被分配的person_id
        """
        cnt = 0 # cnt -> count
        for view in range(self.n_views):
            index = clique.proposal[view]
            if index != -1:
                for i in range(2):
                    _cnt = self.check_kpt_compatibility(view, kps_pair[i], nodes[view][index][i], person_id)
                    if _cnt == -1:
                        return -1
                    else:
                        cnt += _cnt
        return cnt

    def check_kpt_compatibility(self, view, kpt_id, keypoint_index, pid):
        if pid not in self.mpersons_map:
            return -1
        person = self.mpersons_map[pid]
        check_cnt = 0
        if person[kpt_id][view] != -1 and person[kpt_id][view] != keypoint_index:
            # 关节点的id信息冲突了
            return -1

        for paf_id in self.m_kps2paf[kpt_id]:
            check_kps_id = self.paf_dict[paf_id][0] + self.paf_dict[paf_id][1] - kpt_id
            if person[check_kps_id, view] == -1:
                continue
            kps_candidate1 = keypoint_index
            kps_candidate2 = person[check_kps_id, view]
            if kpt_id == self.paf_dict[paf_id][1]:
                kps_candidate1, kps_candidate2 = kps_candidate2, kps_candidate1
            # 两个端点连线的limb的 paf >0
            if self.pafs[view][paf_id][kps_candidate1][kps_candidate2] > 0:
                check_cnt = check_cnt + 1
            else:
                return -1

        for i in range(self.n_views):
            if i == view or person[kpt_id, i] == -1:
                continue
            # 线线距离 也要满足 约束
            if self.m_epi_edges[kpt_id][view][i][keypoint_index,int(person[kpt_id, i])] > 0:
                check_cnt = check_cnt + 1
            else:
                return -1
        return check_cnt

    def push_clique(self, paf_id, proposal):
        if max(proposal[:self.n_views]) == -1:
            return
        clique = Clique(paf_id, proposal)
        clique.score = self.cal_clique_score(clique)
        heapq.heappush(self.cliques, clique)

    def check_person_compatibility_sview(self, master_id, slave_id, view):
        assert master_id < slave_id
        if slave_id < len(self.last_multi_kps3d):
            return -1
        check_cnt = 0
        master = self.mpersons_map[master_id]
        slave = self.mpersons_map[slave_id]

        for kps_id in range(self.n_kps):
            if master[kps_id,view] != -1 and slave[kps_id, view] != -1 and master[kps_id, view] != slave[kps_id, view]:
                return -1

        if master_id < len(self.last_multi_kps3d):
            for kps_id in range(self.n_kps):
                if slave[kps_id, view] != -1:
                    if self.m_temp_edges[kps_id][view][master_id, slave[kps_id][view]] > 0:
                        check_cnt = check_cnt + 1
                    else:
                        return -1

        for paf_id in range(self.n_pafs):
            paf = self.pafs[view][paf_id]
            for candidate in [
                [master[self.paf_dict[paf_id][0], view],slave[self.paf_dict[paf_id][1], view]],
                [slave[self.paf_dict[paf_id][0], view],master[self.paf_dict[paf_id][1], view]]
            ]:
                if min(candidate) >= 0:
                    if paf[candidate[0], candidate[1]] > 0:
                        check_cnt = check_cnt + 1
                    else:
                        return -1
        return check_cnt

    def check_person_compatibility(self, master_id, slave_id):
        assert master_id < slave_id
        if slave_id < len(self.last_multi_kps3d):
            return -1

        check_cnt = 0
        master = self.mpersons_map[master_id]
        slave = self.mpersons_map[slave_id]

        for view in range(self.n_views):
            _check_cnt = self.check_person_compatibility_sview(master_id, slave_id, view)
            if _check_cnt == -1:
                return -1
            else:
                check_cnt += _check_cnt

        for kps_id in range(self.n_kps):
            for view1 in range(self.n_views - 1):
                candidate1_id = master[kps_id, view1]
                if candidate1_id != -1:
                    for view2 in range(view1 + 1, self.n_views):
                        candidate2_id = slave[kps_id, view2]
                        if candidate2_id != -1:
                            if self.m_epi_edges[kps_id][view1][view2][candidate1_id, candidate2_id] > 0:
                                check_cnt += 1
                            else:
                                return -1
        return check_cnt

    def merge_person(self, master_id, slave_id):
        assert master_id < slave_id
        master = self.mpersons_map[master_id]
        slave = self.mpersons_map[slave_id]
        for view in range(self.n_views):
            for kps_id in range(self.n_kps):
                if slave[kps_id, view] != -1:
                    master[kps_id, view] = slave[kps_id, view]
                    self.m_assign_map[view][kps_id][slave[kps_id,view]] = master_id

        self.mpersons_map[master_id] = master
        self.mpersons_map.pop(slave_id)

    def clique2voting(self, clique, voting):
        """
        对clique中的所有bone包含的所有keypoints进行投票
        如果keypoints有person_id对应(也就是assigned!=-1也就是 assigned=person_id)
        则记上一票
        """
        voting.vote = {}
        if len(self.mpersons_map) == 0:
            return voting

        for view in range(self.n_views):
            index = clique.proposal[view]
            if index != -1:
                node = self.m_bone_nodes[clique.paf_id][view][index]
                for i in range(2):
                    assigned = self.m_assign_map[view][self.paf_dict[clique.paf_id][i]][node[i]]
                    if assigned != -1:
                        if assigned not in voting.vote:
                            voting.vote[assigned] = np.zeros(2)

                        voting.vote[assigned][i] += 1
        return voting
