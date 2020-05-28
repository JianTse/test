#coding=utf-8
from flask import Flask,jsonify,request
import cv2
from PIL import Image
from face_recognizer import faceRecognizer
import cv2
import json
import numpy as np
import base64

app = Flask(__name__)#创建一个服务，赋值给APP

faceReconize = faceRecognizer()

@app.route('/ai_faceFeature',methods=['post'])#指定接口访问的路径，支持什么请求方式get，post
def ai_faceFeature():
    client_json = request.json
    if client_json:

        f_read_decode = base64.b64decode(client_json["image"])
        image = np.asarray(bytearray(f_read_decode), dtype="uint8")
        cv_img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # 提取一张人脸的特征
        feat_dict = faceReconize.face_feature(cv_img)
        if feat_dict['flag'] != 'failure':
            feat_dict['feature'] = feat_dict['feature'].tolist()

        # 云端计算结果，返回json
        strJson = json.dumps(feat_dict, ensure_ascii=False)
        print('server return dstJson: ', strJson)
        return strJson
    else:
        json_dict = {}
        json_dict['flag'] = 'failure'
        strJson = json.dumps(json_dict, ensure_ascii=False)
        print(strJson)
        return strJson

@app.route('/ai_faceVerify',methods=['post'])#指定接口访问的路径，支持什么请求方式get，post
def ai_faceVerify():
    client_json = request.json
    if client_json:

        f_read_decode1 = base64.b64decode(client_json["image1"])
        image1 = np.asarray(bytearray(f_read_decode1), dtype="uint8")
        cv_img1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)

        f_read_decode2 = base64.b64decode(client_json["image2"])
        image2 = np.asarray(bytearray(f_read_decode2), dtype="uint8")
        cv_img2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)

        # 两张人脸进行比对，并返回比对结果
        verify_dict = faceReconize.face_verify(cv_img1, cv_img2)

        # 云端计算结果，返回json
        strJson = json.dumps(verify_dict, ensure_ascii=False)
        print('server return dstJson: ', strJson)
        return strJson
    else:
        json_dict = {}
        json_dict['flag'] = 'failure'
        strJson = json.dumps(json_dict, ensure_ascii=False)
        print(strJson)
        return strJson

@app.route('/ai_faceRegister',methods=['post'])#指定接口访问的路径，支持什么请求方式get，post
def ai_faceRegister():
    client_json = request.json
    if client_json:

        f_read_decode = base64.b64decode(client_json["image"])
        image = np.asarray(bytearray(f_read_decode), dtype="uint8")
        cv_img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        faceId = client_json["faceId"]

        register_dict = faceReconize.face_register(cv_img, faceId)

        # 云端计算结果，返回json
        strJson = json.dumps(register_dict, ensure_ascii=False)
        print('server return dstJson: ', strJson)
        return strJson
    else:
        json_dict = {}
        json_dict['flag'] = 'failure'
        strJson = json.dumps(json_dict, ensure_ascii=False)
        print(strJson)
        return strJson

@app.route('/ai_faceIdentify',methods=['post'])#指定接口访问的路径，支持什么请求方式get，post
def ai_faceIdentify():
    client_json = request.json
    if client_json:

        f_read_decode = base64.b64decode(client_json["image"])
        image = np.asarray(bytearray(f_read_decode), dtype="uint8")
        cv_img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        identify_dict = faceReconize.face_identify(cv_img)

        # 云端计算结果，返回json
        strJson = json.dumps(identify_dict, ensure_ascii=False)
        print('server return dstJson: ', strJson)
        return strJson
    else:
        json_dict = {}
        json_dict['flag'] = 'failure'
        strJson = json.dumps(json_dict, ensure_ascii=False)
        print(strJson)
        return strJson

app.run(host='0.0.0.0',port=8802,debug=True)