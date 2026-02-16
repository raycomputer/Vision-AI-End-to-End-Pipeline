from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def run_inference_tuning(model_path, source_img, conf_thresh=0.25, iou_thresh=0.7):
    """
    Inference 분석 및 Threshold 최적화
    Source [3]: Confidence & IoU Threshold Tuning
    """
    
    # 1. 학습된 모델 로드 (Best Weights)
    model = YOLO(model_path)

    # 2. 추론 실행 (Predict)
    # Source [3]: 
    # - conf (Confidence Threshold): 객체라고 확신하는 최소 확률. 높이면 오탐지(FP) 감소.
    # - iou (IoU Threshold): NMS 단계에서 중복 박스를 제거하는 기준.
    results = model.predict(
        source=source_img,
        conf=conf_thresh,
        iou=iou_thresh,
        save=False,       # 별도 저장 로직 사용
        verbose=True
    )

    # 3. 결과 시각화
    for result in results:
        # 결과 이미지를 BGR -> RGB로 변환 (Matplotlib 출력용)
        res_plotted = result.plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(res_rgb)
        plt.title(f"Conf: {conf_thresh} | IoU: {iou_thresh}")
        plt.axis('off')
        plt.show()

        # Source [3]: 오검출 분석 (Confusion Matrix 개념 적용)
        # 로그에서 Detected Classes 개수를 확인하여 False Positive 여부 판단
        print(f"Detected Objects: {len(result.boxes)}")

if __name__ == '__main__':
    # 학습된 모델 경로 지정 (예: runs/detect/train/weights/best.pt)
    model_path = 'yolov8n.pt' # 테스트용으로 pretrained 모델 사용 가능
    test_image = 'https://ultralytics.com/images/bus.jpg' # 테스트 이미지 URL

    # Case 1: Default Setting (Balanced)
    print("--- Running Default Inference ---")
    run_inference_tuning(model_path, test_image, conf_thresh=0.25, iou_thresh=0.7)

    # Case 2: High Precision Mode (엄격한 기준 -> 미탐지 증가 가능성)
    print("--- Running High Precision Mode ---")
    run_inference_tuning(model_path, test_image, conf_thresh=0.7, iou_thresh=0.5)