import base64
import numpy as np
import cv2

def decode_image(base64_string: str) -> np.ndarray:
    """
    Base64 문자열을 OpenCV 이미지(numpy array)로 변환
    Source: Base64 이미지 디코딩 로직 (Base64 -> Buffer -> cv2.imdecode)
    """
    try:
        # 1. Base64 문자열을 바이너리로 디코딩
        img_data = base64.b64decode(base64_string)
        
        # 2. 1차원 배열(Buffer)로 변환
        nparr = np.frombuffer(img_data, np.uint8)
        
        # 3. 이미지 포맷으로 디코딩 (OpenCV)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Image decoding failed")
            
        return img
    except Exception as e:
        # 실제 운영 환경에서는 로깅(Logging) 추가 권장
        raise ValueError(f"Invalid image data: {str(e)}")