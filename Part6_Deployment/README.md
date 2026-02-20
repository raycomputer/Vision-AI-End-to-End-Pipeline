📱 [프론트엔드 / 클라이언트] 
   (test_client.py)
      │
      │ 1️⃣ Request (요청)
      │ "이 사진(Base64 텍스트) 좀 분석해 줘!"
      │
      ▼
🌐 [백엔드 API 서버] 
   (main.py - 0.0.0.0:8000)
      │
      │ 2️⃣ 텍스트를 다시 이미지로 해독
      │ 3️⃣ AI의 뇌 속으로 전달
      │
      ▼
🧠 [AI 비전 모델] 
   (yolov8n.pt 또는 RepViT 가중치)
      │
      │ 4️⃣ "분석 완료! 사람(person)이고 확률은 95%야!"
      │
      ▼
🌐 [백엔드 API 서버] 
   (main.py)
      │
      │ 5️⃣ Response (응답)
      │ 결과를 예쁜 JSON으로 포장해서 반환
      │ {"class_name": "person", "confidence": 0.95}
      │
      ▼
📱 [프론트엔드 / 클라이언트] 
   (test_client.py)
   => ✅ 화면에 "[서버 응답 성공!] 사람 95%" 라고 출력하며 완료!