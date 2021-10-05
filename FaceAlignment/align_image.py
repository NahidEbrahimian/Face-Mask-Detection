import argparse
from .face_alignment import image_align
from .landmarks_detector import LandmarksDetector

# if __name__ == "__main__":
def main(img):

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input image path', default=img, type=str)
    # parser.add_argument('--output', help='output image path', default='output/', type=str)
    parser.add_argument('--landmarks-model-path', help='landmarks model path', default='FaceAlignment/models/shape_predictor_68_face_landmarks.dat', type=str)
    args = parser.parse_args()

    # file_name, file_ext = os.path.splitext(os.path.basename(args.input))
    landmarks_detector = LandmarksDetector(args.landmarks_model_path)
    try:
        all_face_landmarks = landmarks_detector.get_landmarks(args.input)
        for i, face_landmarks in enumerate(all_face_landmarks):
            # image = image_align(args.input, face_landmarks)
            image, bb = image_align(args.input, face_landmarks)
            print(image)
            print(bb)
            return image, bb
            # output_file_path = os.path.join(args.output, file_name + "_" + str(i) + ".jpg")
            # cv2.imwrite(output_file_path, image)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()