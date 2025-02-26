from tensorflow.keras.models import load_model # type: ignore
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# 감정 분석 모델 로드
emotion_model = load_model("models/emotion_model.hdf5", compile=False)

app = FastAPI()

# 감정 분석 함수
def analyze_expression_with_cnn(face_roi):
    try:
        # 얼굴 이미지 전처리 (Grayscale 변환 → 64x64 리사이징 → 정규화)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (64, 64))
        normalized_face = resized_face / 255.0  # 정규화
        reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))  # CNN 입력 형태

        # 감정 예측
        predictions = emotion_model.predict(reshaped_face)
        emotion_label = np.argmax(predictions)  # 가장 높은 확률의 감정 선택

        # 감정 매핑
        emotion_dict = {
            0: "ANGRY",
            1: "DISGUST",
            2: "FEAR",
            3: "HAPPY",
            4: "NEUTRAL",
            5: "SAD",
            6: "SURPRISE"
        }
        return emotion_dict.get(emotion_label, "NEUTRAL")  # 기본 값 = 중립(NEUTRAL)
    
    except Exception as e:
        print(f"Error in analyze_expression_with_cnn: {e}")
        return "NEUTRAL"  # 오류 발생 시 기본 감정 반환

@app.get("/")
def read_root():
    """ 기본 엔드포인트 """
    return {"Hello": "World"}  

@app.post("/analyze")
async def analyze_face(
    image: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...)
):
    """ 클라이언트에서 전송한 이미지와 Bounding Box를 받아 crop 후 표정 분석 """

    try:
        # 이미지 읽기
        image_bytes = await image.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR) # np -> openCV 이미지 형식
        
        if image is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

        # 좌표 정수 변환 및 유효성 검사
        h, w, _ = image.shape
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            return JSONResponse(content={"error": "Invalid bounding box coordinates"}, status_code=400)

        # 얼굴 영역 크롭
        face_roi = image[y1:y2, x1:x2]

        if face_roi.size == 0:
            return JSONResponse(content={"emotion": "NEUTRAL"}, status_code=200)

        # 감정 분석
        emotion = analyze_expression_with_cnn(face_roi)

        return JSONResponse(content=[{"emotion": emotion}], status_code=200)

    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(content={"error": "Server error"}, status_code=500)