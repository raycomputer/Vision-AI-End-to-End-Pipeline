import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # [1] TensorBoard 모니터링
import timm # SOTA 모델 로드를 위한 라이브러리
from tqdm import tqdm
import os

# Part2에서 작성한 Custom Dataset 임포트 가정
# from Part2_DataOps.dataset import CustomDataset 

def train_model(data_dir, num_classes, epochs=30, batch_size=32, lr=1e-4):
    """
    RepViT Fine-tuning 학습 파이프라인
    Source: [1] Transfer Learning Workflow (Load Weights -> Head Replacement -> Fine-tuning)
    """
    
    # 1. Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 2. 모델 로드 (RepViT or EfficientNet 등 SOTA 모델)
    # Source: [1] Pre-trained Weights 로드
    model_name = 'repvit_m1' # timm에서 제공하는 모델명 (예시)
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    except:
        print(f"{model_name} not found, using efficientnet_b0 instead.")
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    model = model.to(device)

    # 3. 데이터 로더 설정 (가정)
    # train_dataset = CustomDataset(os.path.join(data_dir, 'train'), ...)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 4. 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 5. 스케줄러 설정
    # Source: [2] CosineAnnealingLR (학습 후반부 미세 조정을 위해 권장)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 6. TensorBoard 설정
    # Source: [1] 학습 과정의 Loss 및 Accuracy 변화 실시간 시각화
    writer = SummaryWriter(log_dir='runs/repvit_experiment')

    # 7. 학습 루프
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # (실제 실행을 위해서는 train_loader가 필요함)
        # for images, labels in tqdm(train_loader):
        #     images, labels = images.to(device), labels.to(device)
        #
        #     optimizer.zero_grad()
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        #
        #     running_loss += loss.item()
        #     _, predicted = outputs.max(1)
        #     total += labels.size(0)
        #     correct += predicted.eq(labels).sum().item()

        # Scheduler Update
        scheduler.step()

        # Log Metrics (Dummy Data for example)
        # epoch_loss = running_loss / len(train_loader)
        # epoch_acc = correct / total
        # writer.add_scalar('Loss/train', epoch_loss, epoch)
        # writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}] Completed. (Check TensorBoard for details)")

    writer.close()
    
    # 모델 저장
    torch.save(model.state_dict(), "repvit_finetuned.pth")
    print("Model saved!")

if __name__ == "__main__":
    # 실행 예시
    train_model(data_dir='./data', num_classes=3)