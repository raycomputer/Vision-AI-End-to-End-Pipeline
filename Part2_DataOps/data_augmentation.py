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

if __name__ == "__main__":
    import cv2
    import os
    import torch

    # 1. 대상 이미지 경로 설정
    image_path = '../data/cats/cat.jpg'

    # 파일이 실제로 있는지 확인
    if not os.path.exists(image_path):
        print(f"❌ 에러: '{image_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    else:
        print(f"1. 이미지 로드 중... ({image_path})")
        
        # 2. 이미지 읽기 (OpenCV는 기본적으로 BGR로 읽음)
        image = cv2.imread(image_path)
        
        # 3. 색상 채널 변환 (BGR -> RGB)
        # Albumentations와 PyTorch는 RGB 순서를 사용하므로 변환이 필수입니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. 더미(Dummy) 바운딩 박스 생성
        # 작성하신 get_train_transforms 함수에 bbox_params가 포함되어 있어서,
        # 박스 정보가 없으면 에러가 납니다. 테스트를 위해 임의의 값(중앙)을 넣습니다.
        # 포맷: [x_center, y_center, width, height] (0~1 정규화 좌표)
        bboxes = [[0.5, 0.5, 0.1, 0.1]] 
        class_labels = [1] 

        # 5. 변환 함수 호출
        transform = get_train_transforms()
        
        # 6. 실제 변환 수행 (이미지 -> 텐서)
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        # 결과물 추출
        tensor_image = augmented['image']

        # 7. 결과 확인
        print("\n=== 변환 완료 (Tensor 생성) ===")
        print(f"원본 이미지 크기: {image.shape} (H, W, C)")
        print(f"생성된 텐서 shape: {tensor_image.shape} (C, H, W)")
        print(f"텐서 데이터 타입: {tensor_image.dtype}")
        print(f"디바이스 위치: {tensor_image.device}")
        
        print("\n[텐서 데이터 일부 출력 (첫 5개 값)]")
        # 채널별 첫 번째 픽셀 값 확인
        print(tensor_image[:, 0, 0])