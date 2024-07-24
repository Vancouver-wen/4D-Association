import os
import sys

import numpy as np

def convert_redis_format(last_multi_kps3d):
    outputs=[]
    for person_id in last_multi_kps3d:
        output=dict()
        output['id']=person_id
        output['joints19']=(np.array(last_multi_kps3d[person_id])[:,:3]*100).tolist()
        outputs.append(output)
    return str(outputs)