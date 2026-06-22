"""Gradio demo for pneumonia classification and Grad-CAM explanation."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("PNEUMONIA_MODEL_DIR", ROOT / "saved_models"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
MODEL_PATHS = {
    "EfficientNet-B0": MODEL_DIR / "EfficientNet_best.pth",
    "MaxViT-Tiny": MODEL_DIR / "MaxViT_best.pth",
}

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到模型权重：{path}。请按 README 下载权重，或设置 "
            "PNEUMONIA_MODEL_DIR。"
        )
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        return torch.load(path, map_location=DEVICE)


@lru_cache(maxsize=2)
def get_model(name: str) -> nn.Module:
    """Build and cache a model on first use."""
    if name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif name == "MaxViT-Tiny":
        model = timm.create_model(
            "maxvit_tiny_rw_224", pretrained=False, num_classes=2
        )
    else:
        raise ValueError(f"不支持的模型：{name}")

    model.load_state_dict(_load_state_dict(MODEL_PATHS[name]))
    return model.to(DEVICE).eval()


def get_gradcam(
    model: nn.Module, image_tensor: torch.Tensor, class_index: int, model_name: str
) -> np.ndarray:
    """Create a normalized Grad-CAM map and always remove temporary hooks."""
    target_layer = (
        model.features if model_name == "EfficientNet-B0" else model.stages[-1]
    )
    features: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    forward_handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: features.append(output)
    )
    backward_handle = target_layer.register_full_backward_hook(
        lambda _module, _grad_in, grad_out: gradients.append(grad_out[0])
    )

    try:
        output = model(image_tensor)
        model.zero_grad(set_to_none=True)
        output[0, class_index].backward()
        feature_map = features[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1))
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.squeeze().cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()


def apply_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image."""
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def predict(image: Image.Image | None, model_name: str):
    """Return the predicted class, probabilities, and explanation image."""
    if image is None:
        raise gr.Error("请先上传一张胸部 X 光图像。")
    try:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        model = get_model(model_name)
        with torch.inference_mode():
            probabilities = torch.softmax(model(image_tensor), dim=1)[0]
        class_index = int(probabilities.argmax())
        confidence = {
            label: float(probabilities[index].cpu())
            for index, label in enumerate(CLASS_NAMES)
        }
        cam = get_gradcam(model, image_tensor, class_index, model_name)
        return CLASS_NAMES[class_index], confidence, apply_heatmap(np.array(image), cam)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="胸部 X 光肺炎辅助识别") as demo:
        gr.Markdown(
            "# 胸部 X 光肺炎辅助识别\n"
            "上传胸部 X 光片并选择模型，系统将给出分类概率和 Grad-CAM 关注区域。\n\n"
            "> **声明：本项目仅用于研究与教学，不能替代专业医疗诊断。**"
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="胸部 X 光片", type="pil", image_mode="RGB")
                model_choice = gr.Dropdown(
                    choices=list(MODEL_PATHS), value="EfficientNet-B0", label="模型"
                )
                submit = gr.Button("开始分析", variant="primary")
            with gr.Column():
                prediction = gr.Textbox(label="预测结果", interactive=False)
                confidence = gr.Label(label="分类概率", num_top_classes=2)
                explanation = gr.Image(label="Grad-CAM 模型关注区域")
        submit.click(
            predict,
            inputs=(input_image, model_choice),
            outputs=(prediction, confidence, explanation),
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
