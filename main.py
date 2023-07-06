import cv2
from ultralytics import YOLO
import time
import numpy as np
import cv2
from custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback
from tensorflow import keras
import tensorflow as tf
import math
import argparse

WEIGHT = "weights/yolov8s-pose.pt"
KERAS_WEIGHT = "weights/first_weight.h5"
DATASET_NAME = "coco"
# DATASET_NAME = {0: "coke"}
# DATASET_NAME = {0: "coke", 1: "milk", 2: "waterbottle"
# YOLOV8_CONFIG = {"tracker": "botsort.yaml",
#                  "conf": 0.7,
#                  "iou": 0.3,
#                  "show": True,
#                  "verbose": False}

YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7

'''
Command Parameter
{"detect_pose": bool,
 "detect_face": bool}
'''


def process_keypoints(keypoints, conf, frame_width, frame_height, origin=(0, 0)):
    kpts = np.copy(keypoints)
    kpts[:, 0] = (kpts[:, 0] - origin[0]) / frame_width
    kpts[:, 1] = (kpts[:, 1] - origin[1]) / frame_height

    kpts[:, :-1][kpts[:, 2] < conf] = [-1, -1]
    return np.round(kpts[:, :-1].flatten(), 4)


def main():
    HOST = socket.gethostname()
    PORT = 12302

    server = CustomSocket(HOST, PORT)
    server.startServer()

    print("Loading YOLO")
    model = YOLO(WEIGHT, task="pose")
    print("DONE")

    pred_keras = False
    # Limit Keras GPU Usage
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

    while True:
        # Wait for connection from client :}
        conn, addr = server.sock.accept()
        print("Client connected from", addr)

        all_count = dict()
        consec_Count = dict()

        # start = time.time()

        # Process frame received from client
        while True:
            res = dict()
            msg = {"res": res}
            try:
                data = server.recvMsg(
                    conn, has_splitter=True, has_command=True)
                frame_height, frame_width, img, command = data
                detect_pose = command["detect_pose"]
                detect_face = command["detect_face"]

                msg["camera_info"] = [frame_width, frame_height]

                results = model.track(
                    source=img, conf=YOLO_CONF, show=False, verbose=False, persist=True)[0]
                kpts = results.keypoints.cpu().numpy()
                boxes = results.boxes.data.cpu().numpy()

                for person_kpts, person_box in zip(kpts, boxes):
                    person_res = dict()
                    x1, y1, x2, y2 = person_box[:4]
                    x1, y1 = int(max(0, x1)), int(max(0, y1))
                    x2, y2 = int(min(frame_width, x2)), int(
                        min(frame_height, y2))
                    person_id = int(person_box[4])

                    person_res["bbox"] = (x1, y1, x2, y2)
                    person_res["area"] = int((x2-x1) * (y2-y1))
                    person_res["center"] = [
                        int(person_kpts[0, 0]), int(person_kpts[0, 1])]
                    
                    is_inside = all((pose_pt[2] >= KEYPOINTS_CONF for pose_pt in person_kpts[13:15]))
                    person_res["is_inside"] = is_inside

                    if all_count.get(person_id):
                        all_count[person_id] += 1
                    else:
                        all_count[person_id] = 1
                    
                    person_res["alltime_count"] = all_count[person_id]


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
                            rotated_image = cv2.warpAffine(
                                img, rotation_matrix, (frame_width, frame_height))

                            rotated_face = rotated_image[fy1:fy2, fx1:fx2]

                            person_res["faceflatten"] = str(
                                rotated_face.flatten().tolist())[1:-1]
                            person_res["facedim"] = rotated_face.shape[:-1]

                            # cv2.rectangle(rotated_image, (fx1, fy1),
                            #               (fx2, fy2), (255, 0, 200), 2)
                            # cv2.imshow("rot", rotated_face)

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

                    # Draw points
                    # for i, pt in enumerate(person_kpts):
                    #     x, y, p = pt
                    #     if p >= KEYPOINTS_CONF:
                    #         cv2.putText(img, str(i), (int(x), int(
                    #             y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    res[person_id] = person_res

                # Send back result
                # print(res)
                server.sendMsg(conn, json.dumps(msg))

            except Exception as e:
                traceback.print_exc()
                print(e)
                print("Connection Closed")
                del res
                print("Reset YOLO")
                model = YOLO(WEIGHT, task="pose")
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
