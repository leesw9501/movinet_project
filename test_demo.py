import cv2
import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
from movinets.config import _C
import numpy as np
from movinets import MoViNet
import random
import gc


from PIL import Image
import matplotlib.pyplot as plt
import xmltodict
import json
import numpy as np
import gc
import os
import time

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config

from densepose.vis.bounding_box import BoundingBoxVisualizer
from densepose.vis.extractor import create_extractor
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
)

class pose_extractor():
    def __init__(self):
        """
        initalize with config of detectron2 densepose
        """
        config_fpath = "configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        weight_fpath = "densepose_rcnn_R_50_FPN_s1x.pkl"
        self.output_types = ["u", "v"]           # select in [u, v, coarse, fine]

        # setting cfg
        opts = []
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(0.8))
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = weight_fpath
        cfg.freeze()

        # predictor
        self.predictor = DefaultPredictor(cfg)


    def __call__(self, img):
        with torch.no_grad():
            output_dict = dict()
            outputs = self.predictor(img)["instances"]

            # if any objs isn't detected, return empty dict
            if len(outputs) == 0: return output_dict

            # get bnd box
            bnd_outputs = outputs.get("pred_boxes")
            output_dict["pred_boxes"] = bnd_outputs

            # get selected features
            pose_outputs = outputs.get("pred_densepose")
            for key in self.output_types:
                output_dict[key]  = getattr(pose_outputs, key)
            output_dict["instances"] = self.predictor(img)["instances"]

            return output_dict

class SimpleVisualizer():
    def __init__(self, visualizers):
        """
        pick visual kinds that you wank,
        then this instance give the img with visual information
        dp_contour: basic chart spreaded over the body
        dp_segm: segmentation to parts of body
        bbox: bnding box
        """
        vis_dict = {
            "dp_contour": DensePoseResultsContourVisualizer,
            "dp_segm": DensePoseResultsFineSegmentationVisualizer,
            "bbox": BoundingBoxVisualizer,
        }

        self.visualizers = []
        self.extractors = []
        for vis_str in visualizers:
            vis = vis_dict[vis_str]()
            self.visualizers.append(vis)
            self.extractors.append(create_extractor(vis))

    def visualize(self, img_bgr, outputs, img_to_gray=False):
        img = img_bgr
        for i in outputs:
            if i=="instances":
                if img_to_gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data = self._extract_data(outputs["instances"])
                
                for i, visualizer in enumerate(self.visualizers):
                    img = visualizer.visualize(img, data[i])
        return img

    def _extract_data(self, outputs):
        datas = []
        for extractor in self.extractors:
            # TODO plz chk None is ok to place here
            data = extractor(outputs, None)
            datas.append(data)
        return datas

dir_dict=dict()
root = './이상행동 CCTV 영상'
class_list = os.listdir(root)

for class_folder in class_list:
    print(class_folder)
    dir_dict[class_folder]=[]
    place_list = os.listdir(root + '/'+ class_folder)
    for place_num in place_list:
        print(place_num)
        num_list = os.listdir(root + '/'+ class_folder + '/'+ place_num)
        for num in num_list:
            #print(num)
            file_list = os.listdir(root + '/'+ class_folder + '/'+ place_num + '/'+ num)
            for file in file_list:
                #print(file)
                if(file[-4:]=='.mp4'):
                    dir_dict[class_folder].append(root + '/'+ class_folder + '/'+ place_num + '/'+ num + '/'+ file[:-4])
                #print(file[:-4])
                #print(file[-4:]=='.mp4')

class_dict=dict()
class_dict['normal']=0
class_dict['assault']=1
class_dict['fight']=2
class_dict['burglary']=3
class_dict['vandalism']=4
class_dict['swoon']=5
class_dict['wander']=6
class_dict['trespass']=7
class_dict['dump']=8
class_dict['robbery']=9
class_dict['datefight']=10
class_dict['kidnap']=11
class_dict['drunken']=12

class_name=[]
for name in class_dict:
    class_name.append(name)

model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True )
model.classifier[3] = torch.nn.Conv3d(2048, 13, (1,1,1))

model.load_state_dict(torch.load('./model/final_model.pth'))

model.eval()
model.cuda()


for file_path_name in range(12):
    file_path = dir_dict[list(dir_dict.keys())[file_path_name]][0]
    print(file_path)
    num_seqence = 16
    fps_setting = 5
    n_clips = 2
    n_clip_frames = 8
    out_path = './'+str(file_path_name+1)+ '_' +class_name[file_path_name+1]+'_testdemo3_5'

    xml_path = file_path + '.xml'
    # xml 파싱
    f = open(xml_path, 'r')
    read = f.read()
    dict2_type = xmltodict.parse(read)
    
    key_pos = [0, 0]

    if type(dict2_type['annotation']['object']) == type([]):
        for obj in range(len(dict2_type['annotation']['object'])):
            key_pos[0] += int(dict2_type['annotation']['object'][obj]['position']['keypoint']['x'])
            key_pos[1] += int(dict2_type['annotation']['object'][obj]['position']['keypoint']['y'])
        key_pos[0] = int(key_pos[0] / len(dict2_type['annotation']['object']))
        key_pos[1] = int(key_pos[1] / len(dict2_type['annotation']['object']))
    else:
        key_pos[0] += int(dict2_type['annotation']['object']['position']['keypoint']['x'])
        key_pos[1] += int(dict2_type['annotation']['object']['position']['keypoint']['y'])
        
    low = 2160
    width = 3840
    height = 2160
    
    xs = key_pos[0] - int(low / 2)
    xe = key_pos[0] + int(low / 2)
    ys = key_pos[1] - int(low / 2)
    ye = key_pos[1] + int(low / 2)

    if xs < 0:
        xe = xe - xs
        xs = 0
    if ys < 0:
        ye = ye - ys
        ys = 0
    if xe > width:
        xs = xs - (xe - width)
        xe = width
    if ye > height:
        ys = ys - (ye - height)
        ye = height

    st = dict2_type['annotation']['event']['starttime'].split(':')
    du = dict2_type['annotation']['event']['duration'].split(':')

    if len(st) == 3:
        stint = int(((float(st[0]) * 3600) + (float(st[1]) * 60) + float(st[2])) * int(dict2_type['annotation']['header']['fps']))
    else:
        stint = int(((float(st[0]) * 60) + float(st[1])) * int(dict2_type['annotation']['header']['fps']))
    if len(du) == 3:
        etint = stint + int(((float(du[0]) * 3600) + (float(du[1]) * 60) + float(du[2])) * int(dict2_type['annotation']['header']['fps']))
    else:
        etint = stint + int(((float(du[0]) * 60) + float(du[1])) * int(dict2_type['annotation']['header']['fps']))



    video_file = file_path + '.mp4'
    num_seq = num_seqence
    fps_set = fps_setting
    resize_width = 172 #original=3840
    resize_height = 172 #original=2160

    fps_dev = int(30 / fps_set)
    cap = cv2.VideoCapture(video_file)

    out = cv2.VideoWriter(filename=out_path + '.mp4', fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                              fps=fps_setting, frameSize=(resize_width*3, resize_height*3), isColor=True)

    if not out.isOpened():
        print('out File open failed!', out_path + '.mp4')

    # densepose extractor
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    extractor = pose_extractor()
    start_time = time.time()

    #visualizer = SimpleVisualizer(["dp_contour"])

    width = 306
    height = 172


    #key_pos = [int(width/2), int(height/2)]
    add_time = 0
    add_true = 0

    if cap.isOpened():
        i = 0
        img_arr=[]
        pred = [0]
        lastpred = 0
        while True:
            i += 1
            ret, img = cap.read()
            #if i>4700:
            if ret:
                if i % fps_dev == 0:

                    start_time = time.time()

                    if i > stint and i < etint:
                        label = class_dict[dict2_type['annotation']['event']['eventname']]
                    else:
                        label = 0

                    cropped_img = img[ys: ye, xs: xe]
                    #print(img.shape, cropped_img.shape, ys, ye, xs, xe)
                    resized_img = cv2.resize(cropped_img, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)
                    
                    #resized_img = cv2.resize(img, dsize=(306, 172), interpolation=cv2.INTER_AREA)
                    pose_output = extractor(resized_img)
                    #visualizer = SimpleVisualizer(["dp_contour"])

                    #post_img = resized_img
                    visualizer = SimpleVisualizer(["dp_contour"])
                    post_img = visualizer.visualize(resized_img, pose_output)
                    plt.close('all')

                    #cropped_img = post_img[ys: ye, xs: xe]

                    img_arr.append(post_img)
                    if len(img_arr)==num_seqence:
                        data = torch.Tensor(img_arr).reshape(1, 3, 16, 172, 172).float().cuda()/255.
                        #print(data.shape)
                        #print(data)
                        #cv2.imshow("test", img_arr[0])
                        #cv2.waitKey(0)
                        #break

                        with torch.no_grad():
                            model.clean_activation_buffers()
                            for j in range(n_clips):
                                output = model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)])
                            _, pred = torch.max(output, dim=1)                 
                        img_arr=[]
                        #del img_arr[0]

                    end_time = time.time()
                    add_time += end_time-start_time
                    avg_time = add_time / (i / fps_dev)
                    
                    #if pose_output:
                    #    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    resized_img2 = cv2.resize(resized_img, dsize=(172*3, 172*3), interpolation=cv2.INTER_AREA)

                    cv2.putText(resized_img2, "avg time:{:.3f}".format(avg_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.putText(resized_img2, "possible fps:{:.1f}".format(1/avg_time), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.putText(resized_img2, "actual :"+class_name[label], (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                    if pred[0] != 0:
                        cv2.rectangle(resized_img2, (2, 2, 172*3-2, 172*3-2),(0,0,255), 5)
                        cv2.putText(resized_img2, class_name[pred[0]], (380, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        if pred[0] == label:
                            add_true += 1
                            cv2.putText(resized_img2, "predict:"+class_name[pred[0]], (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                        else:
                            cv2.putText(resized_img2, "predict:"+class_name[pred[0]], (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    else:
                        cv2.rectangle(resized_img2, (2, 2, 172*3-2, 172*3-2),(0,255,0), 5)
                        cv2.putText(resized_img2, class_name[pred[0]], (380, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        if pred[0] == label:
                            add_true += 1
                            cv2.putText(resized_img2, "predict:"+class_name[pred[0]], (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                        else:
                            cv2.putText(resized_img2, "predict:"+class_name[pred[0]], (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    cv2.putText(resized_img2, "acc:{:.2f}%".format(100 * add_true / (i / fps_dev)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                    lastpred = pred[0]

                    cv2.imshow("test", resized_img2)
                    cv2.waitKey(1)
                    out.write(resized_img2)

                    #cv2.destroyAllWindows() 
                    #break

            else:
                break

        print(100 * add_true / (i / fps_dev))
        cv2.destroyAllWindows()
        cap.release()
        out.release()

    else:
        print('cannot open the file', out_path + '.mp4')
