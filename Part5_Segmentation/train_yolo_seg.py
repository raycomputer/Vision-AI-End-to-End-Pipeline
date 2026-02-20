from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def train_yolo_segmentation():
    """
    YOLOv8-Seg 학습 파이프라인
    """
    model = YOLO('yolov8n-seg.pt')

    print("--- Starting Segmentation Training ---")
    results = model.train(
        data='coco128-seg.yaml', 
        epochs=10,
        imgsz=640,
        name='yolo_seg_experiment'
    )

def extract_masks_inference(model_path, img_path):
    """
    추론 및 마스크 후처리 (Post-processing)
    """
    model = YOLO(model_path)
    
    # 추론 실행
    results = model.predict(source=img_path, save=False, retina_masks=True)

    for result in results:
        if result.masks is not None:
            # 1. 마스크 데이터 가져오기 (N, H, W 형태)
            masks = result.masks.data.cpu().numpy()

            plt.figure(figsize=(12, 6))
            
            # 2. 원본 이미지 + 오버레이 (RGB 변환 필수)
            plt.subplot(1, 2, 1)
            res_plotted = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            plt.imshow(res_plotted) 
            plt.title("Detection + Segmentation Overlay")

            # 3. Binary Mask 시각화 (차원 에러 해결)
            plt.subplot(1, 2, 2)
            # 모든 객체의 마스크를 한 장의 2차원 이미지로 겹치기
            combined_mask = np.max(masks, axis=0) 
            plt.imshow(combined_mask, cmap='gray')
            plt.title("Extracted Binary Mask (Pixel Level)")
            
            plt.show()

            # 4. 면적 계산 (픽셀 수 카운팅)
            area_pixels = np.sum(combined_mask)
            print(f"✅ 총 객체 차지 면적 (픽셀 수): {area_pixels:.0f} px")
        else:
            print("❌ 이 사진에서는 객체를 찾지 못했습니다.")

if __name__ == "__main__":
    # 1. 학습 실습 (원할 때 주석 해제)
    # train_yolo_segmentation()

    # 2. 추론 및 마스크 추출 실습
    print("--- 모델 다운로드 및 테스트 시작 ---")
    extract_masks_inference('yolov8n-seg.pt', 'https://ultralytics.com/images/bus.jpg')