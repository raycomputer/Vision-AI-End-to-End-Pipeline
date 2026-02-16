from pydantic import BaseModel, Field

# Source: Data Validation (Pydantic 내장)
class InferenceRequest(BaseModel):
    # Base64 문자열은 텍스트이므로 str 타입 정의
    image_base64: str = Field(..., description="Base64 encoded image string")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence Threshold (0.0 ~ 1.0)")

class InferenceResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    # 필요 시 Bounding Box 좌표 추가 가능: bbox: list[float]
