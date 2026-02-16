import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class VisionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        """
        Source [2]: __init__ 구현 (이미지 경로 리스트 로드)
        Args:
            img_dir (str): 이미지 파일 디렉토리
            label_dir (str): YOLO 포맷 txt 라벨 디렉토리
            transform (albumentations.Compose): 적용할 증강 파이프라인
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 이미지 파일 리스트 생성 (.jpg, .png 등)
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        """Source [2]: __len__ 구현 (전체 데이터셋 길이 반환)"""
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        Source [2]: __getitem__ Pipeline
        1. Read (OpenCV) -> 2. Augment -> 3. Normalize & Tensor
        """
        # 1. 파일 경로 설정
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(self.label_dir, img_filename.rsplit('.', 1) + '.txt')

        # 2. 이미지 읽기 (OpenCV는 기본 BGR이므로 RGB로 변환 필수)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 라벨 읽기 (YOLO Format: class_id x_center y_center width height)
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = list(map(float, line.strip().split()))
                    cls_id = int(data)
                    bbox = data[1:] # [x_c, y_c, w, h]
                    bboxes.append(bbox)
                    class_labels.append(cls_id)

        # 4. Augmentation 적용 (Albumentations)
        # 이미지와 바운딩 박스를 동시에 변환
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image'] # Tensor (C, H, W)
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        # 5. 최종 반환 (Target 구성)
        # Note: Detection 학습 시 Batch 처리를 위해서는 별도의 collate_fn이 필요할 수 있음
        target = {}
        target['boxes'] = torch.tensor(bboxes, dtype=torch.float32)
        target['labels'] = torch.tensor(class_labels, dtype=torch.int64)

        return image, target