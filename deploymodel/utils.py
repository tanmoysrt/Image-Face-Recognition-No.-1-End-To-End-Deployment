import pywt
import json
import joblib
import os
import numpy as np
import pandas as pd
import cv2
from deployment.settings import BASE_DIR

face_cascade = cv2.CascadeClassifier(
    str(BASE_DIR)+"/deploymodel"+'/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    str(BASE_DIR)+"/deploymodel"+'/haarcascade_eye.xml')
model_location = str(BASE_DIR)+"/deploymodel"+'/saved_model.pkl'
model = joblib.load(model_location)
with open(str(BASE_DIR)+"/deploymodel"+"./celebrity_key_dict.json") as f:
    celebrities_dic = json.load(f)


def get_cropped_image_if_1_face(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) >= 1:
        for (x, y, w, h) in faces:
            roi_color = img[y:y+h, x:x+w]
            return roi_color
    else:
        print(f'{len(faces)} face are present in image : {path}')
    return None


def wavelet_generaotor(img, mode="haar", level=1):
    inArray = img
    inArray = cv2.cvtColor(inArray, cv2.COLOR_RGB2GRAY)
    inArray = np.float32(inArray)
    # Compute And Process Coefficients
    coeffs = pywt.wavedec2(inArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    # Reconstuction
    inArray_H = pywt.waverec2(coeffs_H, mode)
    inArray_H *= 255
    inArray_H = np.uint8(inArray_H)
    return inArray_H


def predict(img_i):
    img_i = str(BASE_DIR)+"/deploymodel"+'/datatmp/'+img_i
    if os.path.exists(img_i):
        print("Exsists")
    else:
        print("Not Exsists")
    print(img_i)
    img = get_cropped_image_if_1_face(img_i)
    if img is not None:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_wavlet_form = wavelet_generaotor(img, "db1", 5)
        scalled_img_wavlet_form = cv2.resize(img_wavlet_form, (32, 32))
        stacked_image = np.vstack((scalled_raw_img.reshape(
            32*32*3, 1), scalled_img_wavlet_form.reshape(32*32, 1)))
        stacked_image = stacked_image.reshape(1, stacked_image.shape[0])
        key = model.predict(stacked_image)[0]
        for i in celebrities_dic:
            if celebrities_dic[i] == key:
                return i
                break
    else:
        print("Error")
    return None
