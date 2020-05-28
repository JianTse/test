#!/usr/bin/python
# -*- coding: UTF-8 -*-
from model import Backbone, MobileFaceNet, l2_norm
import torch
from torchvision import transforms as trans
import math
import cv2
from mtcnn import MTCNN
from PIL import Image
from utils import findBigFace

class face_learner(object):
    def __init__(self, use_mobilfacenet):

        self.embedding_size = 512
        self.net_depth = 50
        self.drop_ratio = 0.6
        self.net_mode = 'ir_se'  # or 'ir'
        self.threshold = 1.2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.face_detector = MTCNN()

        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if use_mobilfacenet:
            self.model = MobileFaceNet(self.embedding_size).to(self.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(self.net_depth, self.drop_ratio, self.net_mode).to(self.device)
            print('{}_{} model generated'.format(self.net_mode, self.net_depth))
        
        self.threshold = self.threshold
    
    def load_state(self, modelFn):
        self.model.load_state_dict(torch.load(modelFn, map_location=torch.device(self.device)))

    def get_input_by_mtcnn(self, img):
        if (img.shape == (112, 112, 3)):
            pilImg = Image.fromarray(img)
            return pilImg
        else:
            bboxes, landmarks, faces = self.face_detector.align_multi(Image.fromarray(img), limit=None, min_face_size=60)
            if len(bboxes) < 1:
                return None
        idx = findBigFace(bboxes)
        return faces[idx]

    def get_feature(self, pilImg, tta=False):
        if tta:
            mirror = trans.functional.hflip(pilImg)
            emb = self.model(self.test_transform(pilImg).to(self.device).unsqueeze(0))
            emb_mirror = self.model(self.test_transform(mirror).to(self.device).unsqueeze(0))
            embeddings = l2_norm(emb + emb_mirror)
        else:
            embeddings = self.model(self.test_transform(pilImg).to(self.device).unsqueeze(0))
        return embeddings


