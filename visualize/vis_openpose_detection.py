import os
import sys
import random

import numpy as np
import cv2
from tqdm import tqdm
from joblib import Parallel,delayed

from association.openpose_detection import SKEL19DEF
from utils.imageConcat import show_multi_imgs

def each_vis_openpose_detection(
    index,
    image_list,
    openpose_detection_list,
    vis_folder=None,
    return_image=False
    ):
    images=[cv2.imread(image_path) for image_path in image_list]
    for image,openpose_detection in list(zip(images,openpose_detection_list)):
        assert len(openpose_detection.joints)==SKEL19DEF.joint_size
        assert len(openpose_detection.pafs)==SKEL19DEF.paf_size
        for joint_name,joints in list(zip(SKEL19DEF.joint_names,openpose_detection.joints)):
            for joint in joints:
                x,y,score=joint
                cv2.putText(
                    img=image,
                    text=f'{joint_name}:{score:.1f}',
                    org=np.array([x,y],dtype=np.int32),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                    color=(0,0,int(255*score)),
                    thickness=1
                )
                cv2.circle(
                    img=image,
                    center=np.array([x,y],dtype=np.int32),
                    radius=3,
                    color=(0,0,int(255*score)),
                    thickness=-1
                )
        for paf_define,pafs in list(zip(SKEL19DEF.paf_dict,openpose_detection.pafs)):
            jas=openpose_detection.joints[paf_define[0]]
            jbs=openpose_detection.joints[paf_define[1]]
            for index_a,ja in enumerate(jas):
                for index_b,jb in enumerate(jbs):
                    paf=pafs[index_a][index_b]
                    pt1=np.array(ja[:2],dtype=np.int32)
                    pt2=np.array(jb[:2],dtype=np.int32)
                    cv2.line(
                        img=image,
                        pt1=pt1,
                        pt2=pt2,
                        color=(0,int(255*paf),0),
                        thickness=2
                    )
                    center=(pt1+pt2)/2
                    cv2.putText(
                        img=image,
                        text=f"paf:{paf:.1f}",
                        org=center.astype(np.int32),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5,
                        color=(0,int(255*paf),0),
                        thickness=1
                    )
    frame=show_multi_imgs(
        scale=1,
        imglist=images,
        order=(2,3),
    )
    if return_image:
        return frame
    else:
        cv2.imwrite(os.path.join(vis_folder,f"{index:06d}.jpg"),frame)

def vis_openpose_detection(
        image_lists,
        openpose_detection_lists,
        vis_num=300,
        vis_folder="./data/vis_temp"
    ):
    assert len(image_lists)==len(openpose_detection_lists)
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)
    Parallel(n_jobs=-1,backend="threading")(
        delayed(each_vis_openpose_detection)(index,image_list,openpose_detection_list,vis_folder)
        for index,(image_list,openpose_detection_list) in enumerate(tqdm(random.sample(
            list(zip(image_lists,openpose_detection_lists)),
            vis_num
        )))
    )

def main():
    pass

if __name__=="__main__":
    main()