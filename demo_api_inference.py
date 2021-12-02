import os
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle

############# input parameters  #############
from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer
from datetime import datetime

from bodymocap.body_bbox_detector import BodyPoseEstimator
from handmocap.hand_bbox_detector import HandBboxDetector
from integration.copy_and_paste import integration_copy_paste

from demo.demo_frankmocap import run_frank_mocap


class Predict:
  def __init__(self):
      self.args = DemoOptions().parse()
      self.args.use_smplx = True
      self.args.out_dir = "./"

      device = torch.device('cpu')

      self.hand_bbox_detector =  HandBboxDetector('third_view', device)

      #Set Mocap regressor
      self.body_mocap = BodyMocap(self.args.checkpoint_body_smplx, self.args.smpl_dir, device = device, use_smplx= True)
      self.hand_mocap = HandMocap(self.args.checkpoint_hand, self.args.smpl_dir, device = device)
  def predict(self, path):
      self.args.input_path = path
      
      return {"result": run_frank_mocap(self.args, self.hand_bbox_detector, self.body_mocap, self.hand_mocap)}

if __name__ == "__main__":
    predictor = Predict()
    print(predictor.predict("/content/frankmocap/img/"))
