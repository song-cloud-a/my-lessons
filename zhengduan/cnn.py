import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
DATA_DIR = 'data/chest_xray'               # 数据集根目录
MODEL_SAVE_PATH = 'best_model.pth'
IMAGE_SIZE = 224                      # 输入尺寸
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_CLASSES = 2                       # 正常 / 肺炎
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据增强与预处理
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==================== 数据集加载 ====================
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, 'train'),
    transform=train_transforms
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, 'val'),
    transform=val_test_transforms
)
test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, 'test'),
    transform=val_test_transforms
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)

class_names = train_dataset.classes
print(f"类别: {class_names}")
print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(test_dataset)}")

# ==================== 模型定义 ====================
def build_model(num_classes=2):
    # 使用 EfficientNet-B0（也可换成 ResNet50 / DenseNet121）
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # 替换分类头
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

model = build_model(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==================== 训练函数 ====================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# ==================== 测试评估 ====================
def evaluate(model, dataloader):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = running_corrects.double() / len(dataloader.dataset)
    print(f'测试准确率: {accuracy:.4f}')
    return all_preds, all_labels

# ==================== Grad-CAM 可视化 ====================
def get_cam(model, img_tensor, class_idx=None, layer_name='features'):
    """
    生成 Grad-CAM 热力图
    img_tensor: 单张图像 tensor, shape [1, C, H, W]
    layer_name: 要可视化的卷积层名称（EfficientNet 用 'features'）
    """
    model.eval()
    # 注册 hook 获取特征图和梯度
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # 对于 EfficientNet，features 是特征提取部分
    target_layer = dict(model.named_modules())[layer_name]
    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_full_backward_hook(backward_hook)

    # 前向传播
    img_tensor = img_tensor.to(DEVICE)
    output = model(img_tensor)

    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    # 清零梯度并反向传播指定类别
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    # 提取特征图和梯度
    feature_map = features[0].detach()          # [1, C, H, W]
    grad = gradients[0].detach()               # [1, C, H, W]

    # 全局平均池化梯度得到权重
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)  # [1, 1, H, W]

    # ReLU 激活，只保留对类别有正贡献的像素
    cam = torch.relu(cam)

    # 归一化到 [0,1]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # 删除 hook
    hook_forward.remove()
    hook_backward.remove()

    return cam.squeeze().cpu().numpy()

def apply_heatmap(original_img, cam, alpha=0.5):
    """将热力图叠加到原始图像上"""
    # 调整 CAM 到原始图像大小
    cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    # 伪彩色
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # 叠加
    superimposed = heatmap * alpha + original_img
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed

def visualize_gradcam(model, dataloader, num_images=6):
    """从测试集取几张图，生成并保存 Grad-CAM 结果"""
    model.eval()
    os.makedirs('gradcam_results', exist_ok=True)

    # 取一个 batch
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)

    for i in range(min(num_images, inputs.size(0))):
        img_tensor = inputs[i].unsqueeze(0).to(DEVICE)
        label = labels[i].item()

        # 生成 CAM
        cam = get_cam(model, img_tensor, class_idx=None)

        # 还原原始图像（反归一化）
        img_np = inputs[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        original_img = (img_np * 255).astype(np.uint8)

        # 叠加
        superimposed = apply_heatmap(original_img, cam)

        # 保存
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f'Original: {class_names[label]}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(superimposed)
        plt.title(f'Grad-CAM (class: {class_names[label]})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'gradcam_results/sample_{i+1}.png')
        plt.close()
        print(f'样本 {i+1} 热力图已保存')

# ==================== 主程序 ====================
if __name__ == '__main__':
    # 1. 训练
    model = train_model(model, criterion, optimizer, scheduler, EPOCHS)

    # 2. 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'模型已保存至 {MODEL_SAVE_PATH}')

    # 3. 测试评估
    preds, labels = evaluate(model, test_loader)

    # 4. Grad-CAM 可视化
    visualize_gradcam(model, test_loader, num_images=6)