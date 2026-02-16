from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def train_yolo_segmentation():
    """
    YOLOv8-Seg 학습 및 마스크 추출 파이프라인
    Source: YOLO-seg Architecture (Head + Proto Branch) [1]
    Source: Instance Segmentation Data Structure (Polygon) [1]
    """
    
    # 1. 모델 로드 (Segmentation 전용 모델: -seg 접미사 필수)
    # Source: yolov8n-seg.pt 사용 (Nano 버전)
    model = YOLO('yolov8n-seg.pt')

    # 2. 학습 실행 (Training)
    # 데이터셋은 Polygon 좌표(class x1 y1 x2 y2 ...)를 포함해야 함
    print("--- Starting Segmentation Training ---")
    results = model.train(
        data='coco128-seg.yaml', # 예제 데이터셋 (실습용)
        epochs=10,
        imgsz=640,
        name='yolo_seg_experiment'
    )

def extract_masks_inference(model_path, img_path):
    """
    추론 및 마스크 후처리 (Post-processing)
    Source: Matrix Multiplication & Post-processing (Thresholding -> Contours) [1][2]
    """
    model = YOLO(model_path)
    
    # 추론 실행 (RetinaMask 옵션을 켜면 더 높은 품질의 마스크 생성 가능)
    results = model.predict(source=img_path, save=False, retina_masks=True)

    for result in results:
        # 1. 마스크 존재 여부 확인
        if result.masks is not None:
            # Source: Raw Mask Output -> Binary Mask 변환
            # masks.data: (N, H, W) Tensor containing binary masks
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # 첫 번째 객체 마스크 시각화 예시
            plt.figure(figsize=(12, 6))
            
            # 원본 이미지 + Box
            plt.subplot(1, 2, 1)
            plt.imshow(result.plot()) # Ultralytics 기본 시각화
            plt.title("Detection + Segmentation Overlay")

            # Binary Mask만 따로 표시
            plt.subplot(1, 2, 2)
            plt.imshow(masks, cmap='gray')
            plt.title("Extracted Binary Mask (Pixel Level)")
            
            plt.show()

            # Source: Application - 면적 계산 (픽셀 수 카운팅)
            area_pixels = np.sum(masks)
            print(f"Object Area (Pixels): {area_pixels}")

if __name__ == "__main__":
    # 1. 학습 실습 (주석 해제 후 실행)
    # train_yolo_segmentation()

    # 2. 추론 및 마스크 추출 실습
    # 'yolov8n-seg.pt'는 자동으로 다운로드됩니다.
    extract_masks_inference('yolov8n-seg.pt', 'https://ultralytics.com/images/bus.jpg')
