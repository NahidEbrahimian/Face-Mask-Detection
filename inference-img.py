import cv2
import os
import numpy as np
from config import *
from numpy.lib.twodim_base import triu_indices_from
import onnxruntime
from face_detector import FaceDetector
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

model = onnxruntime.InferenceSession("models/mask_detector.onnx", None)

file_name, file_ext = os.path.splitext(os.path.basename(args.input))

img = cv2.imread(args.input)
detection_model = FaceDetector("models/scrfd_500m.onnx")
faces, inference_time, cropped_face = detection_model.inference(img) 

try:
  bboxes = []
  for face in faces:
    face_img = face.cropped_face
    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR )
    face_img = cv2.resize(face_img, (width, height))
    face_img = face_img.astype(np.float32)
    face_img = face_img / 255.0
    face_img = face_img.reshape(1, width, height, 3)

    model_predict = model.run(['dense_1'], {'conv2d_input' : face_img})
    max_index = np.argmax(model_predict)

    if max_index == 0:
      text = "With Mask"
      color = (0, 255, 0)
    else:
      text = "Without Mask"
      color = (0, 0, 255)

    font_size = get_optimal_font_scale(text, (int(face.bbox[3]) - int(face.bbox[1])) / 3)
    cv2.rectangle(img, (int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]), int(face.bbox[3])), color, 2)
    cv2.putText(img, text, (int(face.bbox[0]), int(face.bbox[1])-6), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2,
                    cv2.LINE_AA)

except:
    text = 'Face not Detected'
    font_size = get_optimal_font_scale(text, img.shape[1] // 6)
    cv2.putText(img, text, (int(face.bbox[0]), int(face.bbox[1])-6), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2,
                    cv2.LINE_AA)


# cv2.imshow('Face Mask Detection', img)
cv2.imwrite(os.path.join('output/{}'.format(file_name)+ '.jpg'), img)
cv2.waitKey()