import os
import sys

import numpy as np
import cv2

def line2linedist(pa, raya, pb, rayb):
    """
    直线L1的方向向量为s1，L2的方向向量为s2，点A在直线L1上，点B在直线L2上，
    d=| [s1 s2 AB] | / |s1 x s2|
    [s1 s2 AB]为混合积
    s1 x s2为向量积
    混合积的定义： [a b c]=a*(b x c) i.e. np.multi(a,np.cross(b,c))
    几何解释:
    空间中两条直线的距离为: 两条直线上任意两点在两线直线的法线上的投影
    两条直线的方向向量的叉乘就是法线向量
    """
    if abs(np.vdot(raya, rayb)) < 1e-5:
        return point2linedist(pa, pb, raya)
    else:
        ve = np.cross(raya, rayb)
        ve = ve / np.linalg.norm(ve)
        ve = abs(np.vdot((pa - pb), ve))
        return ve

def point2linedist(pa, pb, ray):
    ve = np.cross(pa - pb, ray)
    return np.linalg.norm(ve)

class Camera():
    def __init__(self, cam_param:dict) -> None:
        super().__init__()
        self.resolution=np.array(cam_param['resolution'])
        self.Ko = np.array(cam_param['K'])
        self.dist=np.array(cam_param['dist'])
        self.R = np.array(cam_param['R'])
        self.t = np.array(cam_param['t'])
        self.alpha=float(cam_param['alpha'])
        # prepare for fast undist image
        self.K,roi=cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.Ko,
            distCoeffs=self.dist,
            imageSize=self.resolution,
            alpha=self.alpha
        )
        self.map1,self.map2=cv2.initUndistortRectifyMap(
            cameraMatrix=self.Ko,
            distCoeffs=self.dist,
            R=None,
            newCameraMatrix=self.K,
            size=self.resolution,
            m1type=cv2.CV_32FC1
        )
        # prepare for fast camera cast ray
        self.Ki = np.linalg.inv(self.K)
        self.Ri_Ki = np.matmul(self.R.T, self.Ki)
        self.Pos = -np.matmul(self.R.T, self.t)

    def cal_ray(self, uv):
        """
        uv是像素平面上的一点(已经去除了畸变)
        K(RX+t)=[u,v,1].T
        X-(-R.T@t)=R.T@K^-1@[u,v,1].T
        注意到R是正定矩阵,也就是 R.T=R^-1
        且 -R.T@t正好是相机在世界坐标系中的位置, X是uv在世界坐标系中的位置
        则 camera cast_ray的方向向量就是 R.T@K^-1@[u,v,1].T
        函数返回归一化的射线方向向量
        """
        var = -self.Ri_Ki.dot(np.append(uv, 1).T)
        return var / np.linalg.norm(var)

    def undistort_image(self,image):
        image_undistort=cv2.remap(image,self.map1,self.map2,cv2.INTER_LINEAR)
        return image_undistort

def convert_to_camera(cam_params):
    return [Camera(cam_param) for cam_param in cam_params]

def test():
    cam_param={
		"K" : 
		[
			[1497.31591796875,0.0,1013.9532470703125],
			[0.0,1496.99462890625,1004.0550537109375],
			[0.0,0.0,1.0]
		],
		"R" : 
		[
			[0.99023151397705078,0.022842187434434891,0.13754963874816895],
			[-0.013483258895576,-0.96617996692657471,0.25751587748527527],
			[0.13877993822097778,-0.25685492157936096,-0.95643383264541626]
		],
		"t" : [0.33267194032669067,0.99631130695343018,2.6735482215881348],
		"dist" : 
		[
			-0.17142131924629211,
			0.11635509133338928,
			-0.00013696192763745785,
			0.00031927728559821844,
			-0.021964715793728828
		],
		"resolution" : [2048,2048],
		"alpha" : 0.0
	}
    Camera(cam_param)

if __name__=="__main__":
    test()