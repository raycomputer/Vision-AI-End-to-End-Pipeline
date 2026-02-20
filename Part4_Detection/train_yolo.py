from ultralytics import YOLO
import os

def train_yolo_model():
    """
    YOLOv8 Custom Data 학습 파이프라인
    """
    
    # 0. 설정 파일 경로 확인
    yaml_path = 'configs/data.yaml'
    if not os.path.exists(yaml_path):
        print(f"❌ 에러: 설정 파일 '{yaml_path}'이 없습니다.")
        print("configs 폴더 안에 data.yaml 파일을 먼저 만들어주세요.")
        return

    print(f"🚀 학습 시작! 설정 파일: {yaml_path}")

    # 1. 모델 로드 (Pre-trained Weights)
    # yolov8n.pt: 가장 가볍고 빠른 모델 (nano 버전)
    # 처음 실행 시 인터넷에서 자동으로 다운로드됩니다.
    model = YOLO('yolov8n.pt') 

    # 2. 학습 실행 (Training)
    results = model.train(
        data=yaml_path,    # 작성한 설정 파일 경로
        epochs=50,         # 학습 횟수
        
        # --- 최적화 설정 ---
        imgsz=640,         # 이미지 크기 (640x640)
        batch=16,          # 배치 사이즈 (메모리 부족하면 8로 줄이세요)
        workers=4,         # 데이터 로딩 속도 (Windows라면 0 권장)
        
        # --- 저장 설정 ---
        project='Part4_Detection/runs', # 결과 저장 폴더 이름
        name='yolo_experiment_1',    # 실험 이름
        exist_ok=True,     # 덮어쓰기 허용
        patience=10,       # 10번 동안 성능 안 오르면 조기 종료

        # freeze=10, # Backbone freeze
        
        device='0' if torch.cuda.is_available() else 'cpu' # GPU 자동 설정
    )

    print(f"✅ 학습 완료! 결과 저장 위치: {results.save_dir}")

if __name__ == '__main__':
    import torch
    # Windows 환경에서 Multiprocessing 오류 방지를 위해 필수
    train_yolo_model()