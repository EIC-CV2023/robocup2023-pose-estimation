import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
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


host = socket.gethostname()
port = 12302

c = CustomSocket(host, port)
c.clientConnect()

cap = cv2.VideoCapture(list_available_cam(10))
cap.set(4, 480)
cap.set(3, 640)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # cv2.imshow('client_cam', frame)

    msg = c.req_with_command(
        frame, {"detect_face": False, "detect_pose": True})

    # print(msg)

    if msg["res"]:
        for person_id, person in msg["res"].items():
            print(person)
            cv2.circle(frame, person["center"], 10, (255,0,0), -1)
            cv2.circle(frame, person["face_center"], 10, (255,255,0), -1)


            if "pose" in person:
                print(person["pose"])
                pose = ["none", "Right", "Left"][person["pose"]]
                cv2.putText(frame, pose, person["center"], cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            if "facedim" in person:
                face_dim = person["facedim"]
                face_img = np.array(person["faceflatten"].split(
                    ", ")).reshape(face_dim + [3]).astype("uint8")
                cv2.imshow("Client face", face_img)

        print(msg)
    cv2.imshow("f", frame)
    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
