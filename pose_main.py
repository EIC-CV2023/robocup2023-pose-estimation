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


def process_keypoints(keypoints, conf, frame_width, frame_height, origin=(0, 0)):
    kpts = np.copy(keypoints)
    kpts[:, 0] = (kpts[:, 0] - origin[0]) / frame_width
    kpts[:, 1] = (kpts[:, 1] - origin[1]) / frame_height

    kpts[:, :-1][kpts[:, 2] < conf] = [-1, -1]
    return np.round(kpts[:, :-1].flatten(), 4)


YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7


def main(detect_pose=False, detect_face=False):
    model = YOLO("weights/yolov8s-pose.pt", task="pose")

    pred_keras = False
    if detect_pose == True:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    "GPU")
                print(len(gpus), "Physical GPUs, ",
                      len(logical_gpus), "Logical GPUs")

            except RuntimeError as e:
                print(e)

            print("Loading Keras Model")
            try:
                keras_model = keras.models.load_model(
                    "weights/first_weight.h5", compile=False)
                pred_keras = True
                print("DONE")
            except:
                print("Error while loading keras model")

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

        results = model.track(source=frame, conf=YOLO_CONF,
                              show=True, verbose=False, persist=True, imgsz=(frame_width, frame_height))[0]
        kpts = results.keypoints.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()
        # print(boxes)
        # print(kpts)

        for person_kpts, person_box in zip(kpts, boxes):
            person_res = dict()
            # print(person_box)
            x1, y1, x2, y2 = person_box[:4]
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(frame_width, x2)), int(min(frame_height, y2))
            person_id = int(person_box[4])
            # print(person_id)

            person_res["bbox"] = (x1, y1, x2, y2)

            # Detect Face
            if detect_face:
                if all((face_pt[2] >= KEYPOINTS_CONF for face_pt in person_kpts[:3])):
                    # print("Face Detect")
                    nose, eye1, eye2, ear1, ear2 = person_kpts[:5, :-1]
                    print(nose, eye1, eye2)
                    face_width = math.sqrt(
                        (ear1[0] - ear2[0])**2 + (ear1[1] - ear2[1])**2)
                    face_height = 1.3 * face_width
                    print(face_width)
                    fx1, fy1 = int(max(nose[0] - face_width / 2, 0)
                                   ), int(max(nose[1] - face_height / 2, 0))
                    fx2, fy2 = int(min(nose[0] + face_width / 2, frame_width)
                                   ), int(min(nose[1] + face_height / 2, frame_height))
                    eye_angle = math.atan2(
                        (eye1[1] - eye2[1]), (eye1[0] - eye2[0])) * 180 / math.pi

                    rotation_matrix = cv2.getRotationMatrix2D(
                        (int(nose[0]), int(nose[1])), eye_angle, 1)
                    # Apply the rotation to the image
                    rotated_image = cv2.warpAffine(
                        frame, rotation_matrix, (frame_width, frame_height))

                    rotated_face = rotated_image[fy1:fy2, fx1:fx2]
                    # print(rotated_face.size[:-1])

                    person_res["faceflatten"] = str(
                        rotated_face.flatten().tolist())[1:-1]

                    person_res["facedim"] = rotated_face.shape[:-1]


                    # cv2.rectangle(rotated_image, (fx1, fy1),
                    #               (fx2, fy2), (255, 0, 200), 2)
                    cv2.imshow("rot", rotated_face)

            if detect_pose:
                if pred_keras:
                    processed_kpts = process_keypoints(
                        person_kpts, KEYPOINTS_CONF, frame_width, frame_height, (x1, y1))
                    pred_pose = np.argmax(keras_model.predict(
                        processed_kpts.reshape((1, 34)), verbose=0), axis=1)
                    # print(processed_kpts)
                    # print(pred_pose[0])
                    person_res["pose"] = int(pred_pose[0])
                else:
                    person_res["pose"] = "NA"

                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (255, 0, 0), 2)

            # Draw points
            for i, pt in enumerate(person_kpts):
                x, y, p = pt
                if p >= KEYPOINTS_CONF:
                    cv2.putText(frame, str(i), (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            res[person_id] = person_res
            del person_res

        # print(res)
        # print(sys.getsizeof(json.dumps(res)))
        # print(len(json.dumps(res).encode("utf-8")))

        cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # print("fps: " + str(round(1 / (time.time() - start), 2)))
        start = time.time()
        # frame2 = np.copy(frame)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detect_pose", action="store_true", help="Detect Pose")
    parser.add_argument(
        "--detect_face", action="store_true", help="Detect Face and Rotate")
    args = parser.parse_args()
    # print(args.detect_pose)

    main(detect_pose=args.detect_pose, detect_face=args.detect_face)
