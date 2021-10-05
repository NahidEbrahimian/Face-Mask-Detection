import argparse
from .face_alignment import image_align
from .landmarks_detector import LandmarksDetector

def main(img):

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', help='input image path', default=img, type=str)
    # parser.add_argument('--output', help='output image path', default='output/', type=str)
    # parser.add_argument('--landmarks-model-path', help='landmarks model path', default='FaceAlignment/models/shape_predictor_68_face_landmarks.dat', type=str)
    # args = parser.parse_args()

    landmarks_detector = LandmarksDetector('FaceAlignment/models/shape_predictor_68_face_landmarks.dat')
    all_face_landmarks = landmarks_detector.get_landmarks(img)

    try:
        for i, face_landmarks in enumerate(all_face_landmarks):
            image, bb = image_align(img, face_landmarks)
            yield image, bb

    except:
        pass
