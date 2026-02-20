import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timm
import os

# 👇 [핵심] 여기에 클래스 순서를 직접 적어주세요! (학습 때와 동일하게)
TARGET_CLASSES = ['cats', 'dogs']  
# ==========================================
# 1. Grad-CAM 클래스 정의
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook 등록
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 2. Backward Pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # 3. Global Average Pooling (GAP)
        gradients = self.gradients
        activations = self.activations
        
        # [수정] 4차원(B, C, H, W)이어야 dim=(2,3)이 가능함
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # 4. Weighted Combination
        cam = torch.sum(weights * activations, dim=1)

        # 5. ReLU & Normalization
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        
        heatmap = cam.cpu().detach().numpy()[0]
        return heatmap

# ==========================================
# 2. 시각화 함수 정의
# ==========================================
def show_cam_on_image(img_path, heatmap, save_path='grad_cam_result.jpg'):
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 찾을 수 없습니다: {img_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap_colored * 0.4 + img_rgb * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # 저장 경로의 폴더가 없으면 생성
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_img)
    print(f"✅ 결과 저장 완료: {save_path}")

# ==========================================
# 3. 실행 코드 (Main)
# ==========================================

if __name__ == "__main__":
    # 1. 설정
    model_name = 'repvit_m1_0'
    # 가중치 파일 경로 (본인 경로에 맞게 수정)
    weights_path = 'results/repvit_finetuned.pth' 
    # 테스트 이미지 경로 (본인 경로에 맞게 수정)
    test_img_path = '../data/test/test_cat2.jpg' 
    
    num_classes = len(TARGET_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 모델 로드
    print(f"🔄 모델 로드 중... ({model_name})")
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        if not os.path.exists(weights_path):
            print(f"❌ 에러: 가중치 파일 '{weights_path}'이 없습니다.")
            exit()
            
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        print("✅ 모델 가중치 복원 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        exit()

    # 3. Target Layer 찾기 (RepViT 차원 에러 해결!)
    target_layer = None
    if 'repvit' in model_name:
        # RepViT에서 차원(H, W)이 살아있는 마지막 Feature Map 위치
        target_layer = model.stages[-1]
    elif 'efficientnet' in model_name:
        target_layer = model.conv_head
    else:
        target_layer = list(model.children())[-2]

    # 4. Grad-CAM 객체 생성
    grad_cam = GradCAM(model, target_layer)

    # 5. 이미지 전처리 & 실행
    if os.path.exists(test_img_path):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        raw_image = cv2.imread(test_img_path)
        if raw_image is None:
            print(f"❌ 이미지를 읽을 수 없습니다: {test_img_path}")
            exit()
            
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=raw_image)['image']
        input_tensor = transformed.unsqueeze(0).to(device)

        # 모델 예측
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = TARGET_CLASSES[pred_idx]
            pred_prob = probs[0][pred_idx].item() * 100

        print(f"\n🔍 모델의 예측 결과:")
        print(f"   👉 클래스: {pred_class} (ID: {pred_idx})")
        print(f"   👉 확신도: {pred_prob:.2f}%")
        
        # Grad-CAM 생성
        print(f"\n📸 Grad-CAM 생성 중... ({pred_class}에 집중)")
        try:
            heatmap = grad_cam(input_tensor, class_idx=pred_idx)
            
            # 결과 저장
            save_name = f"results/grad_cam_{pred_class}.jpg"
            show_cam_on_image(test_img_path, heatmap, save_path=save_name)
        except IndexError as e:
            print("\n❌ [치명적 에러] 차원 문제가 발생했습니다.")
            print(f"이유: Target Layer가 {target_layer}로 잘못 설정되었을 수 있습니다.")
            print(f"에러 메시지: {e}")
        
    else:
        print(f"❌ 테스트 이미지가 없습니다: {test_img_path}")