import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import timm
from tqdm import tqdm
import os
import cv2
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 👇 [수정 1] 내가 원하는 클래스 순서 직접 정의하기
# ==========================================
# 이 리스트의 순서대로 번호가 매겨집니다.
# 0번: cats, 1번: dogs
TARGET_CLASSES = ['cats', 'dogs'] 

class SimpleClassDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # [수정 2] 사용자가 정의한 클래스 리스트 우선 사용
        if target_classes:
            self.classes = target_classes
            print(f"✅ 사용자가 지정한 클래스 순서를 따릅니다: {self.classes}")
        else:
            # 지정 안 하면 기존처럼 폴더 읽어서 자동 정렬
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            print(f"📂 폴더를 읽어 자동으로 클래스를 설정합니다: {self.classes}")

        # 데이터 로드 (지정된 클래스 폴더만 읽음)
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            
            # 폴더가 실제로 있는지 확인
            if not os.path.exists(class_dir):
                print(f"⚠️ 경고: '{class_name}' 폴더가 {root_dir} 안에 없습니다. 스킵합니다.")
                continue

            # jpg, png, jpeg 파일 모두 찾기
            files = glob.glob(os.path.join(class_dir, "*.*"))
            files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for f in files:
                self.image_paths.append(f)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 이미지 읽기 및 예외 처리
        image = cv2.imread(img_path)
        if image is None:
            # 깨진 이미지는 검은색으로 대체 (에러 방지)
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def train_model(data_dir, epochs=5, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 학습 장치: {device}")

    train_dir = os.path.join(data_dir, 'train')
    
    # [수정 3] 데이터셋 생성 시 우리가 정의한 리스트(TARGET_CLASSES) 전달
    dataset = SimpleClassDataset(train_dir, transform=get_transforms(), target_classes=TARGET_CLASSES)
    
    if len(dataset) == 0:
        print("❌ 학습할 이미지가 없습니다. 폴더 경로와 클래스 이름을 확인하세요.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_classes = len(dataset.classes)
    
    print(f"📊 총 이미지: {len(dataset)}장")
    print(f"🎯 클래스 매핑: {dict(zip(range(num_classes), dataset.classes))}") 
    # 출력 예시: {0: 'cats', 1: 'dogs'}

    # RepViT 모델 로드
    model_name = 'repvit_m1_0' 
    print(f"🔄 모델 로드 중... ({model_name})")
    
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    except Exception as e:
        print(f"⚠️ RepViT 로드 실패 ({e}), EfficientNet으로 대체합니다.")
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter(log_dir=f'runs/{model_name}_fixed_class')

    model.train()
    print("\n🚀 학습 시작!")
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({'loss': loss.item()})

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        print(f"   [결과] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.2f}%")

    writer.close()
    torch.save(model.state_dict(), "results/repvit_finetuned.pth")
    print(f"\n✅ 학습 완료! 모델 저장됨.")

if __name__ == "__main__":
    # 데이터 폴더 경로 (./data 안에 train 폴더가 있어야 함)
    train_model(data_dir='../data')