import cv2
import numpy as np
from config import *
# from keras.models import load_model
# from FaceAlignment.align_image import main
import onnxruntime
from face_detector import FaceDetector


def get_optimal_font_scale(text, width):

    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

model = onnxruntime.InferenceSession("models/mask_detector.onnx", None)

# file_name, file_ext = os.path.splitext(os.path.basename(args.input))

detection_model = FaceDetector("models/scrfd_500m.onnx")

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
    faces, inference_time, cropped_face = detection_model.inference(frame) 

    # face_bb = main(frame)

    try:
      bboxes = []
      for face in faces:
        face_img = face.cropped_face
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR )
        face_img = cv2.resize(face_img, (width, height))
        face_img = face_img.astype(np.float32)
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, width, height, 3)

        y_pred = model.run(['dense_1'], {'conv2d_input' : face_img})
        print(y_pred)
        prediction = np.argmax(y_pred)

        if prediction == 0:
          text = f"With Mask, {y_pred[prediction]}"
        else:
          text = "Without Mask, {y_pred[prediction]}"

        font_size = get_optimal_font_scale(text, frame.shape[1] // 6)
        cv2.rectangle(frame, (int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]), int(face.bbox[3])), (144, 0, 0), 2)
        cv2.putText(frame, text, (int(face.bbox[0]), int(face.bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2,
                        cv2.LINE_AA)

    except:
        text = 'Face not Detected'
        font_size = get_optimal_font_scale(text, frame.shape[1] // 6)
        cv2.putText(frame, text, (int(face.bbox[0]), int(face.bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2,
                        cv2.LINE_AA)


    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(10) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('s'):
        cv2.imwrite('FaceMaskDetection.jpg', frame)
        break

video_cap.release()