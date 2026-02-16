from ultralytics import YOLO
import torch

def train_yolo_model():
    """
    YOLOv8 Custom Data 학습 파이프라인
    Source [2]: 학습 실행 및 트러블 슈팅 (Batch Size, AMP, Image Size)
    """
    
    # 1. 모델 로드 (Pre-trained Weights)
    # Source [2]: yolov8n(nano)부터 시작하여 속도와 정확도 트레이드오프 확인
    model = YOLO('yolov8n.pt') 

    # 2. 학습 실행 (Training)
    results = model.train(
        data='configs/data.yaml',  # 작성한 설정 파일 경로
        epochs=50,                 # 학습 횟수
        
        # --- Source [2]: OOM(Out of Memory) 방지 및 최적화 설정 ---
        imgsz=640,      # 입력 이미지 해상도 (메모리 부족 시 512로 축소)
        batch=16,       # 배치 사이즈 (메모리 부족 시 8, 4로 축소)
        amp=True,       # Automatic Mixed Precision (FP16 연산으로 메모리 절약 및 가속)
        workers=4,      # 데이터 로딩 워커 수
        
        # --- Source [2]: 학습 인프라 설정 ---
        device='0',     # Single GPU: '0', Multi-GPU(DDP): '0,1,2,3'
        
        # 프로젝트 관리
        project='vision_ai_project',
        name='yolo_experiment_1',
        exist_ok=True,   # 기존 폴더 덮어쓰기 허용
        patience=10      # Early Stopping (10 epoch 동안 개선 없으면 중단)
    )

    print("Training Completed. Best Model Saved at:", results.save_dir)

if __name__ == '__main__':
    # Windows 환경에서 Multiprocessing 오류 방지를 위해 필수
    train_yolo_model()