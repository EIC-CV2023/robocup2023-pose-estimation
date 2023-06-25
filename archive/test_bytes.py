from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
from tensorflow import keras
import tensorflow as tf
import argparse
import math
import sys
import json


def list_available_cam(max_n):
    list_cam = []
    for n in range(max_n):
        cap = cv2.VideoCapture(n)
        ret, _ = cap.read()

        if ret:
            list_cam.append(n)
        cap.release()

    if len(list_cam) == 1:
        return list_cam[0]
    else:
        print(list_cam)
        return int(input("Cam index: "))


print(b"a")
print("a".encode("utf-8"))
start = time.time()
cap = cv2.VideoCapture(list_available_cam(10))
# cap = cv2.VideoCapture('data/1.mp4')

while cap.isOpened():
    res = dict()
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    frame_height, frame_width = frame.shape[:-1]

    cv2.imshow("frame", frame)
    print("byte", len(frame.tobytes()))
    # print("list", len(" ".join([str(i)
    #       for i in frame.flatten()]).encode("utf-8")))
    # print("list", str(frame.flatten()).encode("utf-8"))
    print(len(json.dumps(frame.flatten().astype("uint8").tolist())))
    print(len(str(frame.flatten().tolist()).encode("utf-8")))

    if cv2.waitKey(1) == ord("q"):
        cap.release()


cv2.destroyAllWindows()
