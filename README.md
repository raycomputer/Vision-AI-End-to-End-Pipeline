# Vision-AI-End-to-End-Pipeline
데이터 구축부터 API 배포까지, Vision AI 파이프라인 실무 첫걸음 강의의 실습 코드입니다.

## 디렉터리 구조
```
Vision-AI-End-to-End-Pipeline/
├── README.md                # 전체 프로젝트 소개 및 실행 가이드
├── requirements.txt         # 전체 프로젝트 공통 의존성 (torch, ultralytics, fastapi 등)
├── Dockerfile               # 배포를 위한 컨테이너 환경 설정
│
├── Part2_DataOps/           # 고품질 데이터셋 구축과 증강
│   ├── data_augmentation.py # Albumentations 활용 증강 스크립트
│   └── custom_dataset.py    # PyTorch Custom Dataset Loader (init, getitem)
│
├── Part3_Classification/    # 효율적인 분류와 XAI
│   ├── train_repvit.py      # RepViT 모델 Fine-tuning 학습 코드
│   └── grad_cam_analysis.py # Grad-CAM 활용 모델 판단 근거 시각화
│
├── Part4_Detection/         # 객체 탐지의 표준, YOLO 마스터
│   ├── configs/
│   │   └── data.yaml        # Custom Data 경로 및 클래스 정의 파일
│   ├── train_yolo.py        # YOLOv8 학습 실행 및 하이퍼파라미터 설정
│   └── inference_tuning.py  # Confidence/IoU Threshold 조절 및 mAP 분석 실습
│
├── Part5_Segmentation/      # 정밀 분석과 Foundation Model
│   ├── train_yolo_seg.py    # YOLO-Seg 학습 및 마스크 추출
│   └── mobile_sam_demo.py   # MobileSAM 활용 Zero-shot 분할 실습
│
├── Part6_Deployment/        # API 서빙 구축 (실무 마무리)
│
├── app/                     # API 서빙 폴더
│   ├── main.py              # FastAPI 서버 인스턴스 및 엔드포인트 정의
│   ├── schemas.py           # Pydantic 데이터 모델 (Request/Response)
│   └── utils.py             # Base64 이미지 디코딩 등 유틸리티
│
└── tests/                   # 테스트 및 성능 측정
    └── locustfile.py        # Locust 부하 테스트 스크립트
```
