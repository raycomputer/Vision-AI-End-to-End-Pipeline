from locust import HttpUser, task, between
import base64

class APIUser(HttpUser):
    # 사용자 간 대기 시간 (1~3초 랜덤)
    wait_time = between(1, 3)

    @task
    def predict_test(self):
        """
        Source: API 성능 테스트 Load Testing (@task 데코레이터)
        """
        # 테스트용 Dummy Base64 이미지 (1x1 Pixel)
        dummy_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        
        payload = {
            "image_base64": dummy_img,
            "threshold": 0.5
        }
        
        # POST 요청 전송
        self.client.post("/predict", json=payload)

