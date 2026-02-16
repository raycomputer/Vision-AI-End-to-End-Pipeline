import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms():
    """
    학습 데이터용 증강 파이프라인 정의
    Source [1]: Geometric Transformations & Pixel-Level Transformations
    Source [2]: Environmental Synthesis (Blur, Noise)
    """
    return A.Compose([
        # --- 1. Geometric Transformations (기하학적 변환) ---
        # 객체의 위치 및 각도 변화에 강인한 모델 학습 목적
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=15, shift_limit=0.1, p=0.5),
        A.Perspective(p=0.2),

        # --- 2. Pixel-Level Transformations (픽셀 단위 변환) ---
        # 조명 변화, 센서 노이즈 등 도메인 변화(Domain Shift) 대응
        A.RandomBrightnessContrast(p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)), # Motion Blur 시뮬레이션
            A.GaussNoise(var_limit=(10.0, 50.0)), # ISO Noise 시뮬레이션
            A.CLAHE(clip_limit=2), # 대비 보정
        ], p=0.3),

        # --- 3. Normalization & Tensor Conversion ---
        # ImageNet 통계값(Mean, Std)으로 정규화 및 PyTorch Tensor 변환
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], 
    # BBoxParams: 데이터 증강 시 바운딩 박스 좌표도 함께 변환되도록 설정 (YOLO 포맷)
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def get_val_transforms():
    """검증 데이터용 변환 (증강 없이 정규화만 수행)"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
