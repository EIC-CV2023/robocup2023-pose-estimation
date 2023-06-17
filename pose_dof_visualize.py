from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
import cv2
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf


JOINT_LINES = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2),
               (0, 3), (5, 6), (4, 7), (0, 7), (1, 6), (2, 5), (3, 4)]
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 4, 7),
         (1, 2, 5, 6), (0, 1, 6, 7), (2, 3, 4, 5)]

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        print(e)


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


def draw_points(frame, keypoints, color=(0, 0, 255)):
    for i, pt in enumerate(keypoints):
        x, y = pt
        cv2.putText(frame, str(i), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_3d_lines(frame, keypoints, joint_list):
    for joint in joint_list:
        cv2.line(frame, [int(k) for k in keypoints[joint[0]]], [
                 int(k) for k in keypoints[joint[1]]], (255, 0, 0), 2)


def get_face_center(keypoints, faces_index_list):
    face_centroid = []
    for face in faces_index_list:
        corner_coord = np.array([keypoints[index] for index in face])
        # print(corner_coord)
        # print(np.mean(corner_coord, axis=0))
        face_centroid.append(np.mean(corner_coord, axis=0))
    return np.array(face_centroid)


def dof(frame, vector_dir, max_len=50, color=(255, 0, 0), center="default"):
    x, y, z = vector_dir
    if center == "default":
        center = int(frame.shape[0] // 2)
        cv2.line(frame, (center, center), (int(center + y * max_len),
                 int(center - z * max_len)), color, 2)
    else:
        cv2.line(frame, center, (int(
            center[0] + y * max_len), int(center[1] - z * max_len)), color, 2)


def draw_dof(frame, vector_out, center="default"):
    v_out = np.reshape(vector_out, (3, 3))
    print(v_out)
    dof(frame, v_out[0], color=(0, 0, 255), center=center)
    dof(frame, v_out[1], color=(0, 255, 0), center=center)
    dof(frame, v_out[2], color=(255, 0, 0), center=center)
    # dof(frame, v_out[3], color=(0,0,128), center=center)
    # dof(frame, v_out[4], color=(0,128,0), center=center)
    # dof(frame, v_out[5], color=(128,0,0), center=center)


def normalize(vectors):
    re_vectors = vectors.reshape((3, 3))
    magnitude = np.linalg.norm((re_vectors), axis=0)
    print(magnitude)
    unit_vector = re_vectors / magnitude
    return unit_vector.flatten()


def process_kpts(xyxy, kpts):
    x1, y1, x2, y2 = xyxy

    copy_kpts = np.copy(kpts)
    copy_kpts[:, 0] = (copy_kpts[:, 0] - x1) / (x2-x1)
    copy_kpts[:, 1] = (copy_kpts[:, 1] - y1) / (y2-y1)

    return copy_kpts.flatten()


model = YOLO("weights/snack-pose.pt", task="pose")
cap = cv2.VideoCapture(list_available_cam(5))
# cap = cv2.VideoCapture("data/test.mov")

keras_model = keras.models.load_model(
    "weights/first_weight.h5", compile=False)

YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7


FRAME_WIDTH = cap.get(3)
FRAME_HEIGHT = cap.get(4)

rand_color_list = np.random.rand(20, 3) * 255
start = time.time()
while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    results = model.track(source=frame, conf=YOLO_CONF,
                          show=False, verbose=False, persist=True)[0]
    kpts = results.keypoints.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    # print(boxes)
    # print(kpts)

    for obj_kpts, obj_box in zip(kpts, boxes):
        # print(obj_box)
        x1, y1, x2, y2 = obj_box[:4]
        cx, cy = x1 + (x2-x1)/2, y1 + (y2-y1)/2
        obj_id = int(obj_box[4])
        print(FRAME_WIDTH, FRAME_HEIGHT)
        # print(obj_id)

        print(obj_kpts.flatten())

        faces_centroid = get_face_center(obj_kpts, FACES)
        # print(faces_centroid)
        draw_points(frame, faces_centroid, (128, 128, 0))

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 2)

        draw_3d_lines(frame, obj_kpts, JOINT_LINES)
        draw_points(frame, obj_kpts)

        keras_input = process_kpts((x1, y1, x2, y2), obj_kpts)

        print(keras_input)

        # pred_dof = normalize(keras_model.predict(process_kpts((x1, y1, x2, y2), obj_kpts)))

        # print(pred_dof.reshape((3,3)))

        # draw_dof(frame, pred_dof, center=(int(cx), int(cy)))

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    start = time.time()

    cv2.imshow("frame", frame)

    key = cv2.waitKey(50)

    if key == ord("q"):
        cap.release()
    if key == ord("s"):
        cv2.imwrite("frame1.jpg", frame)
        cap.release()

cv2.destroyAllWindows()
