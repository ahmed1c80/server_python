from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import face_recognition

app = FastAPI()

@app.post("/detect-face/")
async def detect_face(file: UploadFile = File(...)):
    # قراءة الصورة من الملف
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # تحويل الصورة إلى RGB (لأن face_recognition تستخدم تنسيق RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # الكشف عن الوجوه
    face_locations = face_recognition.face_locations(img_rgb)

    # عدد الوجوه المكتشفة
    return {"faces_detected": len(face_locations), "locations": face_locations}
