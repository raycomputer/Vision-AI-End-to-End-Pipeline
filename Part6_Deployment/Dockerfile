# 1. Base Image (Python 3.9 Slim 버전 권장)
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /code

# 3. 시스템 의존성 설치 (OpenCV 실행을 위해 필수)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 라이브러리 설치
# requirements.txt 파일이 같은 경로에 있다고 가정
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY ./app ./app

# 6. 서버 실행 (Uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
