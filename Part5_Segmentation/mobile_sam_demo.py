from ultralytics import SAM
import numpy as np
import matplotlib.pyplot as plt
import cv2

def run_mobile_sam_demo():
    """
    MobileSAM Zero-shot Segmentation 실습
    Source: Foundation Model & Zero-shot Concept [2]
    Source: Spatial Prompting (Point & Box) [2]
    """
    
    # 1. MobileSAM 모델 로드 (경량화 버전)
    # Source: MobileSAM (Tiny-ViT) - CPU에서도 빠른 추론 가능
    model = SAM('mobile_sam.pt')

    # 테스트 이미지 준비
    img_url = 'https://ultralytics.com/images/zidane.jpg'
    
    print("--- Case 1: Point Prompt (특정 지점 클릭 시뮬레이션) ---")
    # Source: 이미지 내 특정 객체 위에 점(Point) 좌표 입력 -> 전체 영역 마스킹 [2]
    # 예: (900, 370) 좌표에 있는 사람을 분할해라.
    results_point = model.predict(
        source=img_url,
        points=, 
        labels=[3] # 1: Foreground(객체), 0: Background(배경)
    )
    
    # 결과 시각화
    for result in results_point:
        plt.figure(figsize=(10,10))
        plt.imshow(result.plot())
        plt.title("Zero-shot: Point Prompt Result")
        plt.axis('off')
        plt.show()

    print("--- Case 2: Box Prompt (Hybrid Pipeline) ---")
    # Source: 기존 Object Detector로 박스 추출 -> SAM의 프롬프트로 입력 [2]
    # 예: [x1, y1, x2, y2] 박스 영역 안의 객체를 정밀하게 따줘.
    results_box = model.predict(
        source=img_url,
        bboxes= # 예시 박스 좌표
    )

    for result in results_box:
        plt.figure(figsize=(10,10))
        plt.imshow(result.plot())
        plt.title("Zero-shot: Box Prompt Result")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    run_mobile_sam_demo()