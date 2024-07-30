import os
import sys
import argparse
import glob
import copy

import numpy as np
import networkx
import cv2
from loguru import logger
from easydict import EasyDict
from natsort import natsorted
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from joblib import Parallel,delayed

from tools.send2redis import convert_redis_format
from utils.yamlLoader import get_yaml_data
from tools.camera_params import get_camera_params
from association.camera import Camera,convert_to_camera
from tools.get_image_list import get_image_lists
from tools.get_openpose_detection import read_openpose_detections
from visualize.vis_openpose_detection import vis_openpose_detection
from visualize.vis_cam_param_coord import vis_cam_param_coord
from association.graph_construct import GraphConstruct
from association.openpose_detection import SKEL19DEF,OpenposeDetection
from association.graph_associate import GraphAssociate
from association.triangulate_mpersons_map import triangulate_mpersons_map
from utils.FPS import FPS
from visualize.vis_3d_person import vis_3d_person
from openpose.openpose4da import torch_openpose

class FourDagDataset(Dataset):
    def __init__(self,config,skip_num=0,paf_thres=0.4) -> None:
        super().__init__()
        self.cam_params_origin=get_camera_params(
            calibration_json_path=os.path.join('data',config.dataset,'calibration.json')
        )
        self.cam_params=convert_to_camera(self.cam_params_origin)
        self.image_lists=get_image_lists(
            folders=natsorted(filter(lambda x:os.path.isdir(x),glob.glob(os.path.join('data',config.dataset,'video','*'))))
        )
        model_ckpt_path="./openpose/weight/body_25.pth"
        self.openpose=torch_openpose(model_ckpt_path,paf_thres=paf_thres,try_cuda=True)
        self.skip_num=skip_num
    
    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, index):
        if index<self.skip_num: # debug
            return None,None
        image_list=copy.deepcopy(self.image_lists[index])
        openpose_detections=Parallel(n_jobs=1,backend="threading")(
            delayed(self.image_path_to_openpose_detection)(image_path)
            for image_path in image_list
        )
        return image_list,openpose_detections

    def image_path_to_openpose_detection(self,image_path):
        image=cv2.imread(image_path)
        openpose_detection=OpenposeDetection()
        openpose_detection.joints,openpose_detection.pafs=self.openpose.forward( # openpose infer
            oriImg=image,
            scale_search=[0.7,1.0,2.0]
        ) 
        openpose_detection.mapping()
        return openpose_detection

def collect_fn(x):
    return x

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default="./configs/config.yaml")
    args=parser.parse_args()
    config_path=args.config
    config=get_yaml_data(config_path)
    config=EasyDict(config)
    cam_params_origin=get_camera_params(
        calibration_json_path=os.path.join('data',config.dataset,'calibration.json')
    )
    cam_params=convert_to_camera(cam_params_origin)
    image_lists=get_image_lists(
        folders=natsorted(filter(lambda x:os.path.isdir(x),glob.glob(os.path.join('data',config.dataset,'video','*'))))
    )
    openpose_detection_lists=read_openpose_detections(
        txt_paths=natsorted(glob.glob(os.path.join('data',config.dataset,'detection','*.txt')))
    )
    skip_num=0 # debug
    fourDagDataset=FourDagDataset(config,skip_num,paf_thres=0.3)
    fourDagDataloader=DataLoader(
        dataset=fourDagDataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collect_fn
    )

    fps=FPS()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_name="with_temp"
    out = cv2.VideoWriter(f'./myvis/{save_name}.avi',fourcc, 30.0, (5808,2050),True)
    graph_construct=GraphConstruct(
        cameras=cam_params,
        definition=SKEL19DEF
    )
    graph_associate=GraphAssociate(
        n_views=len(cam_params),
        definition=SKEL19DEF
    )
    last_multi_kps3d=dict()
    assert len(image_lists)==len(openpose_detection_lists)
    # read openpose detection result from file
    for frame_index,image_list in enumerate(tqdm(image_lists)):
        openpose_detection_list=openpose_detection_lists[frame_index]
    # infer openpose detection result
    # for frame_index,batch in enumerate(tqdm(fourDagDataloader)):
    #     image_list,openpose_detection_list=batch[0]
        if frame_index<skip_num: # debug
            continue
        fps() # 统计并打印 FPS
        assert len(cam_params)==len(image_list)
        assert len(cam_params)==len(openpose_detection_list)
        # last_multi_kps3d=dict() # 去除时序信息
        graph_4d=graph_construct.__call__(
            kps2d=[openpose_detection.joints for openpose_detection in openpose_detection_list],
            pafs=[openpose_detection.pafs for openpose_detection in openpose_detection_list],
            last_multi_kps3d=last_multi_kps3d
        )
        mpersons_map=graph_associate.__call__(
            kps2d=[openpose_detection.joints for openpose_detection in openpose_detection_list],
            pafs=[openpose_detection.pafs for openpose_detection in openpose_detection_list],
            graph=graph_4d,
            last_multi_kps3d=last_multi_kps3d,
        )
        last_multi_kps3d=triangulate_mpersons_map(
            mpersons_map=mpersons_map,
            cameras=cam_params_origin,
            kpt2ds=list(zip(*[openpose_detection.joints for openpose_detection in openpose_detection_list]))
        )
        frame=vis_3d_person(
            index=frame_index,
            image_list=image_list,
            cam_params=cam_params_origin,
            persons=last_multi_kps3d,
            definition=SKEL19DEF,
            save_name=save_name
        )
        out.write(frame)
    out.release()

if __name__=="__main__":
    main()

# python main.py
# python main.py --config ./configs/config.yaml