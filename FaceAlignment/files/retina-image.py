import cv2
import numpy as np
from config import *
from keras.models import load_model
import matplotlib.pyplot as plt
from retinaface import RetinaFace

model = load_model('weights/model.h5')
image_path = cv2.imread('03.jpg')
faces = RetinaFace.extract_faces(img_path=image_path, align=True)
image = faces[0]
image = cv2.resize(image, (width, height))
image = image / 255.0
img2 = image.reshape(1, width, height, 3)

plt.imshow(image)
# plt.show()

result = model.predict(img2)
prediction = np.argmax(result)

print(result)

if prediction == 0:
  print("With Mask")
  # color = (255, 0, 0)

else:
  print("Without Mask")
  # color = (255, 0, 255)