#coding=utf-8
import cv2
from PIL import Image
from Learner import face_learner
import numpy as np
import glob
import os
import json
from utils import write_bin, read_bin

class faceRecognizer(object):
    def __init__(self):
        self.faceDataDir = './data/facerec/'
        self.modelDir = './data/desired_model/'
        self.modelFn = self.modelDir + 'model_mobilefacenet.pth'
        self.learner = face_learner(use_mobilfacenet=True)
        self.learner.load_state(self.modelFn)
        self.learner.model.eval()
        self.allPersons = []
        self.read_all_register_info()

    def read_all_register_info(self):
        for (path, dirnames, filenames) in os.walk(self.faceDataDir):
            for filename in dirnames:
                personDir = os.path.join(path, filename)
                faceId = filename
                featFn = os.path.join(personDir, 'feat.bin')
                if not os.path.exists(featFn):
                    continue
                info={}
                info['faceId'] = faceId
                info['feature'] = read_bin(featFn)
                self.allPersons.append(info)

    def face_feature(self, cvImg):
        align = self.learner.get_input_by_mtcnn(cvImg)
        retDict = {}
        if align != None:
            embeddings = self.learner.get_feature(align)
            retDict['flag'] = 'successful'
            retDict['feature'] = embeddings[0].data.numpy()
        else:
            retDict['flag'] = 'failure'
            retDict['feature'] = ''
        return retDict

    def face_verify(self, cvImg1, cvImg2):
        retDict = {}
        threshold = 1.5
        retDict['flag'] = 'successful'
        retDict['threshold'] = str(threshold)
        feat_dict1 = self.face_feature(cvImg1)
        feat_dict2 = self.face_feature(cvImg2)
        if feat_dict1['flag'] == 'failure' or feat_dict2['flag'] == 'failure':
            retDict['distant'] = '-1'
            retDict['result'] = 'no face'
        else:
            diff = np.subtract(feat_dict1['feature'], feat_dict2['feature'])
            dist = np.sum(np.square(diff), 0)
            retDict['distant'] = str(dist)
            if dist < threshold:
                retDict['result'] = 'same person'
            else:
                retDict['result'] = 'different person'
        return retDict

    def face_register(self, cvImg, faceId):
        save_path = self.faceDataDir + str(faceId)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        retDict = {}
        featFn = save_path + '/feat.bin'

        feat_dict = self.face_feature(cvImg)
        if feat_dict['flag'] != 'failure':
            write_bin(feat_dict['feature'], featFn)
            retDict['flag'] = 'successful'
        else:
            retDict['flag'] = 'feature'
        imgFn = save_path + '/face.jpg'
        cv2.imwrite(imgFn, cvImg)
        return retDict

    def face_identify(self, cvImg):
        retDict = {}
        threshold = 1.5
        retDict['flag'] = 'successful'
        retDict['threshold'] = str(threshold)
        feat_dict = self.face_feature(cvImg)

        bestId = -1
        minDist = 10000
        if feat_dict['flag'] != 'failure':
            for idx in range(len(self.allPersons)):
                diff = np.subtract(feat_dict['feature'], self.allPersons[idx]['feature'])
                dist = np.sum(np.square(diff), 0)
                if minDist > dist:
                    minDist = dist
                    bestId = idx
        if bestId >= 0:
            retDict['distant'] = str(minDist)
            retDict['faceId'] = str(self.allPersons[bestId]['faceId'])
            retDict['flag'] = 'successful'
        else:
            retDict['flag'] = 'failure'
        return retDict


if __name__ == '__main__':
    faceReconize = faceRecognizer()

    #imgFn1 = 'E:/work/dataSets/face/faceRec/faces_emore/imgs/2/198.jpg'
    #imgFn2 = 'E:/work/dataSets/face/faceRec/faces_emore/imgs/2/200.jpg'
    imgFn1 = 'E:/work/dataSets/face/faceRec/test/318.jpg'
    imgFn2 = 'E:/work/dataSets/face/faceRec/test/628.jpg'
    img1 = cv2.imread(imgFn1)
    img2 = cv2.imread(imgFn2)

    # 人脸注册
    #reg_dict = faceReconize.face_register(img1, 'JanTse')

    # 人脸识别
    identify_dict = faceReconize.face_identify(img2)

    print(identify_dict)

    '''
    # 提取人脸特征
    feat_dict = faceReconize.face_feature(img1)
    if feat_dict['flag'] != 'failure':
        feat_dict['feature'] = feat_dict['feature'].tolist()
    strJson = json.dumps(feat_dict, ensure_ascii=False)
    print('server return dstJson: ', strJson)

    # 人脸比对
    dst_json = faceReconize.face_verify(img1, img2)

    #print('dstJson: ', (dst_json))
    '''

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)