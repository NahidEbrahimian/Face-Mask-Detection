import cv2
import numpy as np
from config import *
from keras.models import load_model
from FaceAlignment.align_image import main

model = load_model("weights/model.h5")

video_cap = cv2.VideoCapture(0)
video_cap.set(cv2.CAP_PROP_FPS, 1)

frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
i = 0

while True:

    rec, frame = video_cap.read()
    if not rec:
        break

    frame_width, frame_height, _ = frame.shape
    face_bb = main(frame)

    try:
        face, bb = next(face_bb)
        i = i + 1
        img = cv2.resize(face, (width, height))
        img = img / 255.0
        img = img.reshape(1, width, height, 3)

        y_pred = model.predict(img)
        prediction = np.argmax(y_pred)

        if prediction == 0:
          y_pred = "With Mask"
        else:
          y_pred = "Without Mask"

        bb = np.array(bb)
        cv2.rectangle(frame, pt1=(bb[0], bb[1]), pt2=(bb[2], bb[3]), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, y_pred, (frame_width // 12, frame_height // 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
    except:
        y_pred = 'Face not Detected'
        cv2.putText(frame, y_pred, (frame_width // 12, frame_height // 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(10) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('s'):
        cv2.imwrite('FaceMaskDetection.jpg', frame)
        break

video_cap.release()