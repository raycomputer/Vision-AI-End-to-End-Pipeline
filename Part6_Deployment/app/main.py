from fastapi import FastAPI, HTTPException
import uvicorn
from app.schemas import InferenceRequest, InferenceResponse
from app.utils import decode_image

# 1. FastAPI 인스턴스 생성 (Source: FastAPI 활용 서버 구축 Boilerplate)
app = FastAPI(
    title="Vision AI Inference Server",
    description="YOLO/RepViT Real-time Inference API",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 경로 (기본값)
    redoc_url="/redoc" # ReDoc 경로
)

# 2. Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server is running"}

# 3. Inference Endpoint
# Source: Inference 엔드포인트 구현 (@app.post, async def)
@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    try:
        # 이미지 디코딩
        image = decode_image(request.image_base64)
        
        # TODO: 모델 추론 로직 연결 (Global Model Load 권장)
        # results = model(image, conf=request.threshold)
        
        # Dummy Logic for Test
        dummy_result = {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95
        }
        return dummy_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    # Source: Performance - 높음 (NodeJS/Go와 유사)
    uvicorn.run(app, host="0.0.0.0", port=8000)
