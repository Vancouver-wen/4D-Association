import os
import sys

import cv2
import numpy as np

def multi_view_triangulate(
        point_2ds,
        poses,
        solve_method="SVD"
    ):
    assert len(point_2ds)==len(poses),"illegal reconstruction parameters"
    if len(poses)<2:
        # triangulation need atleast 2 camera views
        return None
    A=[]
    for point_2d,pose in list(zip(point_2ds,poses)):
        P_matrix=np.concatenate(
            (np.array(pose['R']),np.expand_dims(np.array(pose['t']).T,axis=1)),
            axis=1
        )
        x=point_2d[0]
        y=point_2d[1]
        A.append(x*P_matrix[2]-P_matrix[0])
        A.append(y*P_matrix[2]-P_matrix[1])
    A=np.array(A).astype(np.float32)
    if solve_method=="SVD":
        U,sigma,VH = np.linalg.svd(A,full_matrices=True)
        vector=VH[-1]
        point_3d=vector[:3]/vector[3]
    else:
        # 这个解法不好
        eigen_value,eigen_vector = np.linalg.eig(A.T@A)
        vector=eigen_vector[np.argmin(eigen_value)]
        point_3d=vector[:3]/vector[3]
    return point_3d

def easy_multi_view_triangulate(
        point_2ds,
        poses,
        solve_method="SVD"
    ):
    if len(point_2ds)<2:
        return None
    normalized_point_2ds=[]
    for point_2d,pose in list(zip(point_2ds,poses)):
        temp=cv2.undistortPoints(
            src=np.expand_dims(np.array(point_2d,dtype=np.float32),axis=0),
            cameraMatrix=np.array(pose['K'],dtype=np.float32),
            distCoeffs=np.squeeze(np.array(pose['dist'],dtype=np.float32)),
            # P=np.array(camera1['K'])
        )
        normalized_point_2ds.append(np.squeeze(temp))
    point_3d=multi_view_triangulate(
        point_2ds=normalized_point_2ds,
        poses=poses,
        solve_method=solve_method
    )
    return point_3d

def triangulate_mpersons_map(mpersons_map,cameras,kpt2ds):
    multi_kps3d_map=dict()
    for key in mpersons_map:
        mperson=mpersons_map[key]
        multi_kps3d_map[key]=[]
        assert len(mperson)==len(kpt2ds)
        for mjoint,kpt2d in list(zip(mperson,kpt2ds)):
            point_2ds,poses=[],[]
            for index,point_2d,pose in list(zip(mjoint,kpt2d,cameras)):
                if index<0:
                    continue
                point_2ds.append(point_2d[index][:2])
                poses.append(pose)
            point_3d=easy_multi_view_triangulate(
                point_2ds=point_2ds,
                poses=poses
            )
            if point_3d is None:
                point_3d=[0,0,0,-1]
            else:
                point_3d=point_3d.tolist()+[1]
            multi_kps3d_map[key].append(point_3d)
    return multi_kps3d_map
            