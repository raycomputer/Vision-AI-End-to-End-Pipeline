import requests
import base64
import os

def test_inference_server():
    # 1. 접속할 서버 주소 (방금 띄우신 FastAPI 주소)
    url = "http://0.0.0.0:8000/predict"
    
    # 2. 테스트할 이미지 경로 (본인 컴퓨터에 있는 사진 경로로 수정하세요!)
    image_path = "/Users/doyeonjung/한빛앤/Vision-AI-End-to-End-Pipeline/Part4_Detection/datasets/images/test/2_jpg.rf.c839c333e069e5c3ebb9c457194d2983.jpg" 

    if not os.path.exists(image_path):
        print(f"❌ 에러: 이미지를 찾을 수 없습니다 -> {image_path}")
        return

    print(f"이미지 '{image_path}'를 변환하는 중...")

    # 3. 이미지를 열어서 컴퓨터가 읽을 수 있는 텍스트(Base64)로 변환
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # 4. 서버(Swagger UI)가 기다리고 있는 JSON 규칙에 맞게 포장
    # 서버 화면에서 봤던 "image_base64"와 "threshold"를 그대로 적어줍니다.
    payload = {
        "image_base64": encoded_string,
        "threshold": 0.5
    }

    # 5. 서버로 POST 요청 쏘기!
    print("🚀 서버로 데이터를 전송합니다...")
    try:
        response = requests.post(url, json=payload)
        
        # 6. 결과 확인
        if response.status_code == 200:
            print("\n✅ [서버 응답 성공!]")
            print(response.json()) # 서버가 찾아낸 결과(좌표, 클래스 등) 출력
        else:
            print(f"\n❌ [서버 응답 에러] 상태 코드: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ 에러: 서버에 연결할 수 없습니다. 0.0.0.0:8000 서버가 켜져 있는지 확인하세요.")

if __name__ == "__main__":
    test_inference_server()