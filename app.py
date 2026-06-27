import os
import uuid
import cv2
import numpy as np
import onnxruntime
from config import *
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from face_detector import FaceDetector

app = FastAPI(title="Face Mask Detection API")

mask_model = onnxruntime.InferenceSession("models/mask_detector.onnx", None)
detection_model = FaceDetector("models/scrfd_500m.onnx")

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        text_size = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=scale / 10,
            thickness=1
        )
        new_width = text_size[0][0]
        if new_width <= width:
            return scale / 10
    return 1

def process_image(img):
    faces, inference_time, cropped_face = detection_model.inference(img)

    results = []

    try:
        for face in faces:
            face_img = face.cropped_face
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            face_img = cv2.resize(face_img, (width, height))
            face_img = face_img.astype(np.float32)
            face_img = face_img / 255.0
            face_img = face_img.reshape(1, width, height, 3)

            model_predict = mask_model.run(['dense_1'], {'conv2d_input': face_img})
            max_index = np.argmax(model_predict)

            if max_index == 0:
                text = "With Mask"
                color = (0, 255, 0)
            else:
                text = "Without Mask"
                color = (0, 0, 255)

            bbox = [
                int(face.bbox[0]),
                int(face.bbox[1]),
                int(face.bbox[2]),
                int(face.bbox[3]),
            ]

            font_size = get_optimal_font_scale(text, (bbox[3] - bbox[1]) / 3)

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(
                img,
                text,
                (bbox[0], bbox[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                color,
                2,
                cv2.LINE_AA
            )

            results.append({
                "label": text,
                "bbox": bbox
            })

    except Exception as e:
        text = "Face not Detected"
        font_size = get_optimal_font_scale(text, img.shape[1] // 6)
        cv2.putText(
            img,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        return img, {
            "message": "Face not Detected",
            "detections": [],
            "error": str(e)
        }

    return img, {
        "message": "success",
        "detections": results
    }


@app.get("/")
def home():
    return {"message": "Face Mask Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    processed_img, result = process_image(img)

    os.makedirs("output", exist_ok=True)
    output_filename = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join("output", output_filename)

    cv2.imwrite(output_path, processed_img)

    return {
        "result": result,
        "output_image": f"/output/{output_filename}"
    }


@app.get("/output/{filename}")
def get_output_image(filename: str):
    file_path = os.path.join("output", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(file_path, media_type="image/jpeg", filename=filename)
