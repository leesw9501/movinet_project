import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import xmltodict
import json
import numpy as np
import gc
import os

"""
python script implemented by HJ based on 전처리mp4만들기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ >> dense pose implementation
for visual >> if you erase comments of this lines, you can visualize bnding box in video
"""

dir_dict=dict()
#root = './이상행동 CCTV 영상'
root = './test_dataset'
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



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class pose feature extractor
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
        data = self._extract_data(outputs["instances"])

        if img_to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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


def mk_mp4_label(file_path, out_path, num_seqence=16, fps_setting=5):
    xml_path = file_path + '.xml'
    video_file = file_path + '.mp4'
    num_seq = num_seqence
    fps_set = fps_setting
    resize_width = 800
    resize_height = 800

    # xml 파싱
    f = open(xml_path, 'r')
    read = f.read()
    dict2_type = xmltodict.parse(read)

    label = dict()
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

    st = dict2_type['annotation']['event']['starttime'].split(':')
    du = dict2_type['annotation']['event']['duration'].split(':')

    if len(st) == 3:
        stint = int(((float(st[0]) * 3600) + (float(st[1]) * 60) + float(st[2])) * int(
            dict2_type['annotation']['header']['fps']))
    else:
        stint = int(((float(st[0]) * 60) + float(st[1])) * int(dict2_type['annotation']['header']['fps']))
    if len(du) == 3:
        etint = stint + int(((float(du[0]) * 3600) + (float(du[1]) * 60) + float(du[2])) * int(
            dict2_type['annotation']['header']['fps']))
    else:
        etint = stint + int(((float(du[0]) * 60) + float(du[1])) * int(dict2_type['annotation']['header']['fps']))

    # 크롭 및 리사이즈
    width = int(dict2_type['annotation']['size']['width'])
    height = int(dict2_type['annotation']['size']['height'])
    depth = int(dict2_type['annotation']['size']['depth'])

    if width > height:
        low = height
    else:
        low = width

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

    start_end = []
    for obj in label:
        # print(obj)
        for act in label[obj]:
            # print(act)
            for se in label[obj][act]:
                # print(se)
                start_end.append([act, se[0], se[1]])

    fps_dev = int(int(dict2_type['annotation']['header']['fps']) / fps_set)
    cap = cv2.VideoCapture(video_file)

    label = []
    cla = class_dict[dict2_type['annotation']['event']['eventname']]
    i = 0

    out = cv2.VideoWriter(filename=out_path + '.mp4', fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                          fps=fps_setting, frameSize=(resize_width, resize_height), isColor=True)
    if not out.isOpened():
        print('out File open failed!', out_path + '.mp4')

    # densepose extractor
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    extractor = pose_extractor()

    # for visulization
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)

    if cap.isOpened():
        while True:
            i += 1
            ret, img = cap.read()
            if ret:
                if i % fps_dev == 0:
                    cropped_img = img[ys: ye, xs: xe]
                    resized_img = cv2.resize(cropped_img, dsize=(resize_width, resize_height),
                                             interpolation=cv2.INTER_AREA)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    pose_output = extractor(img)

                    visualizer = SimpleVisualizer(["dp_contour"])
                    post_img = visualizer.visualize(img, pose_output)

                    # for visual
                    # if pose_output:
                    #     print(pose_output["pred_boxes"])
                    #     for idx in range(len(pose_output["pred_boxes"])):
                    #         xyxy = pose_output["pred_boxes"].tensor[idx]
                    #         cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])),
                    #                       (18, 127, 15), 5)
                    # cv2.imshow("test", img)
                    # out.write(resized_img)
                    # if cv2.waitKey(1) == 27:
                    #     break

                    if i > stint and i < etint:
                        label.append(cla)
                    else:
                        label.append(0)

                if i % 100 == 0:
                    gc.collect()
            else:
                break

        cap.release()
        out.release()

    else:
        print('cannot open the file', out_path + '.mp4')

    label2 = np.array(label, dtype=np.uint8)

    np.save(out_path + '.npy', label2)
    print('success', out_path)



print(dir_dict)
for _, path in dir_dict.items():
    mk_mp4_label(file_path = path[0], out_path = './iterdata/', num_seqence=16, fps_setting=5)