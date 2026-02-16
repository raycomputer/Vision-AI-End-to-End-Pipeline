import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    Source: [2] 모델이 이미지를 보는 영역 역추적 (Backprop -> Gradient Calculation -> GAP)
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook 등록: Forward 및 Backward 시 Feature Map과 Gradient를 캡처
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 2. Backward Pass (Gradient Calculation)
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # 3. Global Average Pooling (GAP) of Gradients -> Weights Alpha
        # Source: [2] Weight alpha = Mean(Gradients)
        pooled_gradients = torch.mean(self.gradients, dim=[3, 4])

        # 4. Weighted Combination of Feature Maps
        # Source: [2] Linear Combination (Weights * Feature Maps)
        activation = self.activations
        for i in range(activation.shape):
            activation[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()

        # 5. ReLU & Normalization
        heatmap = np.maximum(heatmap, 0) # ReLU
        heatmap /= torch.max(torch.tensor(heatmap)) # Normalize 0~1

        return heatmap

def show_cam_on_image(img_path, heatmap):
    """
    Source: [2] Heatmap과 원본 이미지 Overlay 시각화
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) # 모델 입력 크기에 맞춤
    
    # Heatmap 크기 조정 및 컬러맵 적용
    heatmap = cv2.resize(heatmap, (img.shape[5], img.shape))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 이미지 합성 (Weighted Sum)
    superimposed_img = heatmap * 0.4 + img * 0.6
    
    # 결과 저장 및 출력
    cv2.imwrite('grad_cam_result.jpg', superimposed_img)
    print("Grad-CAM result saved as 'grad_cam_result.jpg'")

# 사용 예시 (pseudo-code)
# model = ... (학습된 모델 로드)
# target_layer = model.features[-1] # CNN의 마지막 레이어 타겟팅
# grad_cam = GradCAM(model, target_layer)
# heatmap = grad_cam(input_tensor)
# show_cam_on_image('test.jpg', heatmap)