import albumentations as A
import cv2
import matplotlib.pyplot as plt

# 시각화 파이프라인
def get_visualize_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Perspective(p=0.2),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.GaussNoise(var_limit=(10.0, 50.0)),
        ], p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 1. 실제 이미지 불러오기
image_path = 'data/cat.jpg' # <-- 여기에 이미지 파일명을 넣으세요!
image = cv2.imread(image_path)

if image is None:
    print(f"이미지를 찾을 수 없습니다: {image_path}")
    print("같은 폴더에 jpg 파일을 넣고 파일명을 맞춰주세요.")
else:
    # OpenCV는 BGR로 읽으므로 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. 가짜 바운딩 박스 (테스트용)
    bboxes = [[0.5, 0.5, 0.2, 0.2]]
    class_labels = [1]

    # 3. 변환 적용
    transform = get_visualize_transforms()
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_image = augmented['image']

    # 4. 시각화 (원본 vs 변환본 비교)
    plt.figure(figsize=(10, 5))

    # 원본
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    # 변환본 (증강 적용)
    plt.subplot(1, 2, 2)
    plt.title("Augmented Image")
    plt.imshow(aug_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show() # 창이 뜨면서 이미지가 보입니다.