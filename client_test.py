#coding=utf-8
import json
import base64
import requests
import cv2
import random

sever_local = 'http://127.0.0.1:8802'

def sendOneImgs(imgFn):
    '''
    get one face feature
    imgFn: face image
    return: face feature
    '''
    with open(imgFn, mode='rb') as file:
        img = file.read()

    data = {}
    data['image'] = base64.encodebytes(img).decode("utf-8")

    url = sever_local + "/ai_faceFeature"
    result = requests.request("POST", url, json=data)
    print(result.text)  # 打印算法返回结果

def sendPairImgs(imgFn1, imgFn2):
    '''
    compare two face is same person
    imgFn1, imgFn: two face image file name
    return: compare result
    '''
    with open(imgFn1, mode='rb') as file:
        img1 = file.read()
    with open(imgFn2, mode='rb') as file:
        img2 = file.read()

    data = {}
    data['image1'] = base64.encodebytes(img1).decode("utf-8")
    data['image2'] = base64.encodebytes(img2).decode("utf-8")

    url = url = sever_local + "/ai_faceVerify"
    result = requests.request("POST", url, json=data)
    print(result.text)  # 打印算法返回结果
    # print('\n')

'''
imgFn1 = 'E:/work/dataSets/face/faceRec/test/318.jpg'
imgFn2 = 'E:/work/dataSets/face/faceRec/test/628.jpg'

sendPairImgs(imgFn1, imgFn2)

# 显示上传图像
img1 = cv2.imread(imgFn1)
img2 = cv2.imread(imgFn2)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey(0)
'''

def read_lfw_pairs():
    lfw_img_dir = 'E:/work/dataSets/face/faceRec/lfw/'
    lfw_fn = 'E:/work/dataSets/face/faceRec/lfw_pairs/pairsDevTest.txt'

    file = open(lfw_fn)
    content = file.readlines()
    file.close()

    while(1):
        idx = random.randint(0,len(content))
        params = content[idx].split()
        if len(params) == 3:   #same
            imgFn = lfw_img_dir + '/' + params[0] + '/' + params[0]
            imgFn1 = imgFn + '_%04d.jpg' % (int(params[1]))
            imgFn2 = imgFn + '_%04d.jpg' % (int(params[2]))
            isSame = 1
            print('groundTruth: same person')
        else:
            imgFn1 = lfw_img_dir + '/' + params[0] + '/' + params[0] + '_%04d.jpg' % (int(params[1]))
            imgFn2 = lfw_img_dir + '/' + params[2] + '/' + params[2] + '_%04d.jpg' % (int(params[3]))
            isSame = 0
            print('groundTruth: different person')

        #sendOneImgs(imgFn1)
        sendPairImgs(imgFn1, imgFn2)

        img1 = cv2.imread(imgFn1)
        img2 = cv2.imread(imgFn2)
        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)
        cv2.waitKey(0)

read_lfw_pairs()


