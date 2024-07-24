import os
import sys

import numpy as np
import cv2

from utils.imageConcat import show_multi_imgs
from association.camera import Camera

def vis_cam_param_coord(
        cam_params:list[Camera],
        image_list:list[str],
        vis_path:str="./data/vis_temp.jpg"
    ):
    images=[cv2.imread(image_path) for image_path in image_list]
    origin=np.array([0,0,0],dtype=np.double)
    xcoord=np.array([1,0,0],dtype=np.double)
    ycoord=np.array([0,1,0],dtype=np.double)
    zcoord=np.array([0,0,1],dtype=np.double)
    for image,cam_param in list(zip(images,cam_params)):
        point2d,_=cv2.projectPoints(
            objectPoints=np.stack([origin,xcoord,ycoord,zcoord]),
            rvec=cam_param.R,
            tvec=cam_param.t,
            cameraMatrix=cam_param.Ko,
            distCoeffs=cam_param.dist
        )
        point2d=point2d.squeeze().astype(np.int32)
        cv2.line(
            img=image,
            pt1=point2d[0],
            pt2=point2d[1],
            color=(255,0,0),
            thickness=2
        )
        cv2.line(
            img=image,
            pt1=point2d[0],
            pt2=point2d[2],
            color=(0,255,0),
            thickness=2
        )
        cv2.line(
            img=image,
            pt1=point2d[0],
            pt2=point2d[3],
            color=(0,0,255),
            thickness=2
        )
    frame=show_multi_imgs(
        scale=1,
        imglist=images,
        order=(2,3)
    )
    cv2.imwrite(vis_path,frame)
        

def main():
    pass

if __name__=="__main__":
    main()