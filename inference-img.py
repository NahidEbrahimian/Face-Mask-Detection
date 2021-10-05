import cv2
import os
import numpy as np
from config import *
from keras.models import load_model
from FaceAlignment.align_image import main
import argparse

def get_optimal_font_scale(text, width):

    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

parser = argparse.ArgumentParser()
parser.add_argument("--input", default='input/02.jpg', type=str)
args = parser.parse_args()

model = load_model("weights/model.h5")

file_name, file_ext = os.path.splitext(os.path.basename(args.input))

img = cv2.imread(args.input)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face_BB = main(img)

try:
    face, BB = next(face_BB)
    face_img = cv2.resize(face, (width, height))
    face_img = face_img / 255.0
    face_img = face_img.reshape(1, width, height, 3)

    y_pred = model.predict(face_img)
    prediction = np.argmax(y_pred)

    if prediction == 0:
      y_pred = "With Mask"
    else:
      y_pred = "Without Mask"

    BB = np.array(BB)
    font_size = get_optimal_font_scale(y_pred, img_rgb.shape[1] // 4)
    cv2.rectangle(img, pt1=(BB[0], BB[1]), pt2=(BB[2], BB[3]), color=(0, 255, 0),thickness=2)
    cv2.putText(img, y_pred, (img_rgb.shape[0] // 12, img_rgb.shape[1] // 12), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2,
                    cv2.LINE_AA)

except:
    y_pred = 'Face not Detected'
    font_size = get_optimal_font_scale(y_pred, img_rgb.shape[1] // 4)
    cv2.putText(img, y_pred, (img_rgb.shape[0] // 12, img_rgb.shape[1] // 12), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2,
                    cv2.LINE_AA)


cv2.imshow('Face Mask Detection', img)
cv2.imwrite(os.path.join('output/{}'.format(file_name)+ '.jpg'), img)
cv2.waitKey()