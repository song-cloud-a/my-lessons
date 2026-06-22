import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import timm

# -------------------- 配置 --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# 模型权重文件路径
MODEL_PATHS = {
    'EfficientNet-B0': 'saved_models/EfficientNet_best.pth',
    'MaxViT-Tiny':'saved_models/MaxViT_best.pth'
}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------- 模型加载函数 --------------------
def load_efficientnet():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATHS['EfficientNet-B0'], map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_maxvit():
    model = timm.create_model('maxvit_tiny_rw_224', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATHS['MaxViT-Tiny'], map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 缓存已加载的模型，避免重复加载
loaded_models = {}

def get_model(name):
    if name not in loaded_models:
        if name == 'EfficientNet-B0':
            loaded_models[name] = load_efficientnet()
        elif name == 'MaxViT-Tiny':
            loaded_models[name] = load_maxvit()
    return loaded_models[name]

# -------------------- Grad-CAM 生成 --------------------
def get_gradcam(model, img_tensor, class_idx, model_name):
    """根据模型名称选择目标层并生成 Grad-CAM"""
    if model_name == 'EfficientNet-B0':
        target_layer = model.features
    elif model_name == 'MaxViT-Tiny':
        target_layer = model.stages[-1]
    else:
        raise ValueError('不支持的模型')

    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_full_backward_hook(backward_hook)

    img_tensor = img_tensor.to(DEVICE)
    output = model(img_tensor)
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    feature_map = features[0].detach()      # [1, C, H, W]
    grad = gradients[0].detach()            # [1, C, H, W]
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    hook_forward.remove()
    hook_backward.remove()
    return cam.squeeze().cpu().numpy()

def apply_heatmap(original_img, cam, alpha=0.5):
    cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = heatmap * alpha + original_img
    return np.clip(superimposed, 0, 255).astype(np.uint8)

# -------------------- 预测函数 --------------------
def predict(image, model_name):
    if image is None:
        return "请上传图片", {}, None

    model = get_model(model_name)
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    # 生成 Grad-CAM
    cam = get_gradcam(model, img_tensor, class_idx=pred_idx, model_name=model_name)
    original_np = np.array(image.convert('RGB'))
    superimposed_img = apply_heatmap(original_np, cam)

    return pred_class, confidences, superimposed_img

# -------------------- 简洁界面 --------------------
with gr.Blocks(title="肺炎X光辅助诊断（双模型）") as demo:
    gr.Markdown("## 🩻 肺炎X光影像辅助诊断系统")
    gr.Markdown("上传胸部X光片，选择模型，系统将给出诊断结果及模型关注区域。")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="上传胸部 X 光片", type="pil", image_mode="RGB")
            model_choice = gr.Dropdown(
                choices=['EfficientNet-B0', 'MaxViT-Tiny'],
                value='EfficientNet-B0',
                label="选择诊断模型"
            )
            btn = gr.Button("开始分析", variant="primary")

        with gr.Column(scale=1):
            pred_label = gr.Textbox(label="预测结果", interactive=False)
            conf_output = gr.Label(label="置信度", num_top_classes=2)
            cam_output = gr.Image(label="模型关注区域 (Grad-CAM)")

    btn.click(
        fn=predict,
        inputs=[input_img, model_choice],
        outputs=[pred_label, conf_output, cam_output]
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)