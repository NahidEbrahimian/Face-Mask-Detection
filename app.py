import os
import uuid
import cv2
import numpy as np
import onnxruntime
import mediapipe as mp
from config import *
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(title="Face Mask Detection API")

mask_model = onnxruntime.InferenceSession("models/mask_detector.onnx", None)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

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
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_mp = face_detection.process(img_rgb)

    results = []

    try:
        if not results_mp.detections:
            raise Exception("No face detected in the image.")
        i=0
        for detection in results_mp.detections:
            bbox_raw = detection.location_data.relative_bounding_box
            xmin = int(bbox_raw.xmin * w)
            ymin = int(bbox_raw.ymin * h)
            width_box = int(bbox_raw.width * w)
            height_box = int(bbox_raw.height * h)
            
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmin + width_box), min(h, ymin + height_box)

            face_img = img[ymin:ymax, xmin:xmax]
            
            if face_img.size == 0:
                continue

            cv2.imwrite(f"FACE{i}.jpg", face_img)
            i+=1
            face_img_resized = cv2.resize(face_img, (width, height))
            face_img_resized = face_img_resized.astype(np.float32)
            face_img_resized = face_img_resized / 255.0
            face_img_resized = face_img_resized.reshape(1, width, height, 3)
            
            model_predict = mask_model.run(['dense_1'], {'conv2d_input': face_img_resized})
            max_index = np.argmax(model_predict)
            print(model_predict, max_index)

            if max_index == 0:
                text = "With Mask"
                color = (0, 255, 0)
            else:
                text = "Without Mask"
                color = (0, 0, 255)

            bbox = [xmin, ymin, xmax, ymax]

            font_size = get_optimal_font_scale(text, width_box / 3)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                img,
                text,
                (xmin, ymin - 10),
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
        font_size = get_optimal_font_scale(text, w // 6)
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
    return {"message": "Face Mask Detection API (MediaPipe) is running"}


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
