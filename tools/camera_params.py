import os
import sys
import json

import numpy as np
from natsort import natsorted

def get_camera_params(calibration_json_path):
    with open(calibration_json_path,'r') as f:
        calibrations=json.load(f)
    cam_params=[]
    for key in natsorted(calibrations.keys()):
        cam_param=dict()
        cam_param['K']=np.array(calibrations[key]['K']).squeeze().reshape((3,3)).tolist()
        cam_param['dist']=np.array(calibrations[key]['distCoeff']).squeeze().tolist()
        cam_param['R']=np.array(calibrations[key]['R']).squeeze().reshape((3,3)).tolist()
        cam_param['t']=np.array(calibrations[key]['T']).squeeze().tolist()
        cam_param['resolution']=np.array(calibrations[key]['imgSize']).squeeze().tolist()
        cam_param['alpha']=calibrations[key]['rectifyAlpha']
        cam_params.append(cam_param)
    return cam_params