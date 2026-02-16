# Vision-AI-End-to-End-Pipeline
데이터 구축부터 API 배포까지, Vision AI 파이프라인 실무 첫걸음 강의의 실습 코드입니다.

# 디렉터리 구조
Vision-AI-End-to-End-Pipeline/
├── README.md                  # 전체 프로젝트 소개 및 실행 가이드
├── requirements.txt           # 전체 프로젝트 공통 의존성 (torch, ultralytics, fastapi 등) [2]
│
├── Part2_DataOps/             # 고품질 데이터셋 구축과 증강 [3]
│   ├── data_augmentation.py   # Albumentations 활용 증강 스크립트 [3]
│   └── custom_dataset.py      # PyTorch Custom Dataset Loader (__init__, __getitem__) [4]
│
├── Part3_Classification/      # 효율적인 분류와 XAI [3]
│   ├── train_repvit.py        # RepViT 모델 Fine-tuning 학습 코드 [3]
│   └── grad_cam_analysis.py   # Grad-CAM 활용 모델 판단 근거 시각화 [3]
│
├── Part4_Detection/           # 객체 탐지의 표준, YOLO 마스터 [4]
│   ├── configs/
│   │   └── data.yaml          # Custom Data 경로 및 클래스 정의 파일 [4]
│   ├── train_yolo.py          # YOLOv8 학습 실행 및 하이퍼파라미터 설정 [5]
│   └── inference_tuning.py    # Confidence/IoU Threshold 조절 및 mAP 분석 실습 [5]
│
├── Part5_Segmentation/        # 정밀 분석과 Foundation Model [5]
│   ├── train_yolo_seg.py      # YOLO-Seg 학습 및 마스크 추출 [6]
│   └── mobile_sam_demo.py     # MobileSAM 활용 Zero-shot 분할(Point/Box Prompt) 실습 [7]
│
└── Part6_Deployment/          # API 서빙 구축 (실무 마무리) [7]
    ├── app/
    │   ├── main.py            # FastAPI 서버 인스턴스 및 엔드포인트 정의 [2]
    │   ├── schemas.py         # Pydantic 데이터 모델 (Request/Response) [2]
    │   └── utils.py           # Base64 이미지 디코딩 등 유틸리티 [2]
    ├── tests/
    │   └── locustfile.py      # Locust 부하 테스트 스크립트 [2]
    └── Dockerfile             # 배포를 위한 컨테이너 환경 설정 [2]