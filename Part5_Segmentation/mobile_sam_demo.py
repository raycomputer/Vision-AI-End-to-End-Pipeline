from ultralytics import SAM
import numpy as np
import matplotlib.pyplot as plt
import cv2

def run_mobile_sam_demo():
    """
    MobileSAM Zero-shot Segmentation 실습
    """
    
    # 1. MobileSAM 모델 로드 (파일이 없으면 자동으로 다운로드됩니다)
    model = SAM('mobile_sam.pt')

    # 테스트 이미지 준비 (유명한 지단 사진)
    img_url = 'https://ultralytics.com/images/zidane.jpg'
    
    print("--- Case 1: Point Prompt (특정 지점 클릭 시뮬레이션) ---")
    # (900, 370) 좌표는 이미지 오른쪽 인물의 얼굴/몸 근처입니다.
    results_point = model.predict(
        source=img_url,
        points=[[900, 370]],  # [[x, y]] 형태로 입력
        labels=[1]            # 1: Foreground(찾고 싶은 객체), 0: Background
    )
    
    # 결과 시각화
    for result in results_point:
        plt.figure(figsize=(10,10))
        plt.imshow(result.plot())
        plt.title("Zero-shot: Point Prompt Result")
        plt.axis('off')
        plt.show()

    print("--- Case 2: Box Prompt (Hybrid Pipeline) ---")
    # 지단 사진의 왼쪽 인물을 감싸는 박스 좌표 예시입니다.
    # [x1, y1, x2, y2] 형식
    results_box = model.predict(
        source=img_url,
        bboxes=[[439, 437, 824, 710]] 
    )

    for result in results_box:
        plt.figure(figsize=(10,10))
        plt.imshow(result.plot())
        plt.title("Zero-shot: Box Prompt Result")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    run_mobile_sam_demo()