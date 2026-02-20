import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

# 이전에 만든 transforms 함수가 필요하므로 임시로 여기에 정의하거나 import 해야 합니다.
# (테스트를 위해 간단한 버전을 여기에 넣었습니다)
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.Resize(224, 224), # 테스트를 위해 크기 고정
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


class VisionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        """
        Args:
            img_dir (str): 이미지 파일 디렉토리
            label_dir (str): YOLO 포맷 txt 라벨 디렉토리
            transform (albumentations.Compose): 적용할 증강 파이프라인
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 이미지 파일 리스트 생성 (.jpg, .png 등)
        # 파일이 없는 경우를 대비해 빈 리스트 처리
        if os.path.exists(img_dir):
            self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        else:
            self.img_files = []
            print(f"경고: {img_dir} 폴더가 없습니다.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. 파일 경로 설정
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # [수정 1] 파일명에서 확장자 제거 후 .txt 붙이기
        # 예: cat.jpg -> cat -> cat.txt
        filename_without_ext = img_filename.rsplit('.', 1)[0]
        label_path = os.path.join(self.label_dir, filename_without_ext + '.txt')

        # 2. 이미지 읽기
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 라벨 읽기 (YOLO Format)
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = list(map(float, line.strip().split()))
                    if len(data) >= 5: # 데이터가 온전한지 확인
                        # [수정 2] 리스트의 첫 번째 요소를 가져와야 함 (data -> data[0])
                        cls_id = int(data[0]) 
                        bbox = data[1:] # [x_c, y_c, w, h]
                        bboxes.append(bbox)
                        class_labels.append(cls_id)
        
        # 4. Augmentation 적용
        if self.transform:
            # 박스가 없을 때 에러 방지를 위해 예외 처리 혹은 더미 데이터가 필요할 수 있음
            if len(bboxes) == 0:
                # 박스가 없으면 이미지 변환만 수행 (bbox_params가 있으면 에러날 수 있음 주의)
                # 여기서는 편의상 박스가 있다고 가정하거나 빈 리스트 처리
                pass 
            
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                image = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
            except Exception as e:
                print(f"Augmentation Error at {img_filename}: {e}")

        # 5. 최종 반환
        target = {}
        target['boxes'] = torch.tensor(bboxes, dtype=torch.float32)
        target['labels'] = torch.tensor(class_labels, dtype=torch.int64)

        return image, target

if __name__ == "__main__":
    # 1. 테스트 환경 설정 (경로 확인)
    # 현재 폴더 구조가 data/cat.jpg 라고 가정
    base_dir = '../data/cats' # 이미지가 있는 폴더명
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"'{base_dir}' 폴더를 생성했습니다. 여기에 이미지를 넣어주세요.")
    
    # 테스트용 라벨 파일 생성 (cat.txt) - 이미지가 있다면
    # cat.jpg가 있다고 가정하고, 같은 이름의 txt 파일을 만듭니다.
    label_file = os.path.join(base_dir, 'cat.txt')
    if not os.path.exists(label_file):
        with open(label_file, 'w') as f:
            # 클래스ID 1, 중심좌표(0.5, 0.5), 너비높이(0.2, 0.2)
            f.write("1 0.5 0.5 0.2 0.2\n")
        print(f"테스트용 라벨 파일 생성: {label_file}")

    print("--- 데이터셋 테스트 시작 ---")

    # 2. 데이터셋 인스턴스 생성
    dataset = VisionDataset(
        img_dir=base_dir, 
        label_dir=base_dir, # 이미지와 라벨이 같은 폴더에 있다고 가정
        transform=get_train_transforms()
    )

    # 3. 데이터 잘 불러오는지 확인
    if len(dataset) > 0:
        print(f"데이터셋 크기: {len(dataset)}개")
        
        # 첫 번째 데이터 가져오기 (__getitem__ 호출)
        img, target = dataset[0]
        
        print("\n=== 첫 번째 데이터 결과 ===")
        print(f"이미지 텐서 Shape: {img.shape} (C, H, W)")
        print(f"이미지 텐서 타입: {img.dtype}")
        print(f"Min/Max 값: {img.min():.2f} ~ {img.max():.2f}")
        
        print("\n--- Target 정보 ---")
        print(f"Boxes: \n{target['boxes']}")
        print(f"Labels: {target['labels']}")
        print("=========================")
        print("성공! 데이터셋 클래스가 정상 작동합니다.")
    else:
        print(f"❌ '{base_dir}' 폴더에 .jpg 이미지가 없습니다. 이미지를 넣고 다시 실행해주세요.")