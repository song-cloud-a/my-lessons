import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as tv_models
import timm
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import warnings
import gc
warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
DATA_DIR = 'data/chest_xray'                        # 数据集根目录
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224                                    # 输入图像统一尺寸
BATCH_SIZE = 16                                     # 批大小
EPOCHS = 25                                         # 训练总轮数
LR = 1e-4                                           # 学习率
WEIGHT_DECAY = 1e-5                                 # 权重衰减（L2正则）
NUM_CLASSES = 2                                     # 正常与肺炎两类
MODEL_SAVE_DIR = 'saved_models'                     # 模型保存路径
GRADCAM_DIR = 'gradcam_results'                     # Grad-CAM 单模型结果
GRADCAM_COMPARE_DIR = 'gradcam_comparison'          # 多模型 Grad-CAM 对比
WRONG_DIR = 'wrong_predictions'                     # 错分样本保存路径
CSV_REPORT = 'results.csv'
TEXT_REPORT = 'report.txt'
# 创建所需目录
for d in [MODEL_SAVE_DIR, GRADCAM_DIR, GRADCAM_COMPARE_DIR, WRONG_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 数据增强 ====================
train_transform = create_transform(
    input_size=IMAGE_SIZE, is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1',            # 自动搜索最优增强策略
    re_prob=0.25, re_mode='pixel', re_count=1,      # 随机擦除
    mean=(0.485, 0.456, 0.406),                     # ImageNet 均值
    std=(0.229, 0.224, 0.225)                       # ImageNet 标准差
)
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ==================== 数据加载 ====================
def get_dataloaders(batch_size=BATCH_SIZE):
    """加载训练/验证/测试数据集，返回 dataloader 和类别名"""
    train_ds = ImageFolder(os.path.join(DATA_DIR,'train'), transform=train_transform)
    val_ds = ImageFolder(os.path.join(DATA_DIR,'val'), transform=val_transform)
    test_ds = ImageFolder(os.path.join(DATA_DIR,'test'), transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes

# ==================== 模型构建 ====================
def build_model(model_name='efficientnet', pretrained=True):
    """根据名称构建模型，支持 EfficientNet-B0 和 MaxViT-Tiny"""
    if model_name == 'efficientnet':
        model = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)  # 替换分类头
    elif model_name == 'maxvit':
        model = timm.create_model('maxvit_tiny_rw_224', pretrained=pretrained, num_classes=NUM_CLASSES)
    else:
        raise ValueError('未知模型')
    return model.to(DEVICE)

# ==================== 训练与评估 ====================
def train_one_epoch(model, loader, optimizer, criterion, mixup_fn=None, ema_model=None):
    # 单轮训练，支持 MixUp/CutMix 及指数移动平均(EMA)
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)   # 混合增强
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if ema_model: ema_model.update(model)             # 更新 EMA 模型
        total_loss += loss.item() * images.size(0)
        # 计算准确率（MixUp 后标签是 soft label）
        if targets.dim() > 1:
            _, labels = targets.max(1)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
        else:
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    """评估模型，返回损失、准确率、预测值、真实标签、正类概率"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    if criterion is None: criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, all_probs = [], [], []
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        all_probs.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())  # 肺炎类概率
    acc = correct / total
    return total_loss / total, acc, all_preds, all_labels, all_probs

# ==================== Grad-CAM 可视化 ====================
def get_gradcam(model, image_tensor, layer_name=None):
    """
    生成 Grad-CAM 热力图
    layer_name: 若不指定，自动选择最后一个卷积层
    """
    model.eval()
    image_tensor = image_tensor.to(DEVICE)
    feature_maps, gradients = [], []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # 自动寻找最后一个卷积层
    if layer_name is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                layer_name = name
                break
    target_layer = dict(model.named_modules())[layer_name]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = torch.argmax(output, dim=1).item()
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    fmap = feature_maps[0].detach()
    grad = gradients[0].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)   # 全局平均池化梯度
    cam = (weights * fmap).sum(1)                    # 加权求和
    cam = torch.relu(cam)                            # 只保留正贡献
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化
    fh.remove()
    bh.remove()
    return cam.squeeze().cpu().numpy()

def apply_heatmap(img, cam, alpha=0.5):
    """将热力图叠加到原图"""
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.clip(heatmap * alpha + img, 0, 255).astype(np.uint8)

def batch_gradcam(model, loader, class_names, save_dir=GRADCAM_DIR, num_images=6):
    """批量生成单模型 Grad-CAM 结果并保存"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    images_saved = 0
    for imgs, labels in loader:
        for i in range(len(imgs)):
            img_tensor = imgs[i].unsqueeze(0)
            cam = get_gradcam(model, img_tensor)
            # 反归一化恢复原图
            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            heatmap_img = apply_heatmap(img_np, cam)
            plt.imsave(os.path.join(save_dir, f'sample_{images_saved+1}_{class_names[labels[i]]}.png'), heatmap_img)
            images_saved += 1
            if images_saved >= num_images:
                return

def compare_gradcam(models_dict, loader, class_names, save_dir=GRADCAM_COMPARE_DIR, num_images=6):
    """生成多模型 Grad-CAM 并排对比图"""
    os.makedirs(save_dir, exist_ok=True)
    models_dict = {k: v.eval() for k, v in models_dict.items()}
    images_saved = 0
    for imgs, labels in loader:
        for i in range(len(imgs)):
            fig, axes = plt.subplots(1, len(models_dict)+1, figsize=(4*(len(models_dict)+1), 4))
            img_tensor = imgs[i].unsqueeze(0)
            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original: {class_names[labels[i]]}')
            axes[0].axis('off')
            for j, (name, model) in enumerate(models_dict.items()):
                cam = get_gradcam(model, img_tensor)
                heatmap_img = apply_heatmap(img_np, cam)
                axes[j+1].imshow(heatmap_img)
                axes[j+1].set_title(name)
                axes[j+1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{images_saved+1}.png'))
            plt.close()
            images_saved += 1
            if images_saved >= num_images:
                print(f'对比 Grad-CAM 图已保存至 {save_dir}')
                return

# ==================== 主程序 ====================
if __name__ == '__main__':
    # 1. 加载数据
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    # 2. 初始化两个模型
    eff_model = build_model('efficientnet', pretrained=True)
    max_model = build_model('maxvit', pretrained=True)
    models_dict = {'EfficientNet': eff_model, 'MaxViT': max_model}

    # 3. 分别训练和评估两个模型
    for name, model in models_dict.items():
        print(f"\n开始训练 {name}...")
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = SoftTargetCrossEntropy()                          # 配合 MixUp 使用
        ema_model = ModelEmaV2(model, decay=0.9998, device=DEVICE)   # 指数移动平均，稳定推理
        mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5,
                         label_smoothing=0.1, num_classes=NUM_CLASSES)

        best_acc = 0.0
        train_losses, val_losses, val_accs = [], [], []              # 记录训练曲线

        for epoch in range(EPOCHS):
            t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, mixup_fn, ema_model)
            val_criterion = nn.CrossEntropyLoss()
            v_loss, v_acc, _, _, _ = evaluate(ema_model.module, val_loader, val_criterion)
            scheduler.step()
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            val_accs.append(v_acc)
            print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss {t_loss:.4f} Acc {t_acc:.4f} | Val Loss {v_loss:.4f} Acc {v_acc:.4f}')
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(ema_model.module.state_dict(), os.path.join(MODEL_SAVE_DIR, f'{name}_best.pth'))

        # 加载最佳模型并在测试集评估
        best_model_path = os.path.join(MODEL_SAVE_DIR, f'{name}_best.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        _, test_acc, all_preds, all_labels, all_probs = evaluate(model, test_loader, nn.CrossEntropyLoss())
        print(f'{name} 测试准确率: {test_acc:.4f}')

        # 保存训练/验证曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend(); plt.title(f'{name} Loss')
        plt.subplot(1, 2, 2)
        plt.plot(val_accs, label='Val Accuracy', color='green')
        plt.legend(); plt.title(f'{name} Val Accuracy')
        plt.tight_layout()
        plt.savefig(f'{name}_training_curves.png')
        plt.close()

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.savefig(f'{name}_confusion_matrix.png'); plt.close()

        # ROC 曲线
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], '--')
        plt.legend()
        plt.savefig(f'{name}_roc_curve.png'); plt.close()

        # 结果导出
        df = pd.DataFrame({'pred': all_preds, 'label': all_labels, 'prob': all_probs})
        df.to_csv(f'{name}_results.csv', index=False)
        report = classification_report(all_labels, all_preds, target_names=class_names)
        with open(f'{name}_report.txt', 'w') as f:
            f.write(report)

    # 4. 生成 Grad-CAM 可视化（单模型与对比）
    batch_gradcam(eff_model, test_loader, class_names, save_dir=GRADCAM_DIR, num_images=6)
    batch_gradcam(max_model, test_loader, class_names, save_dir=GRADCAM_DIR, num_images=6)
    compare_gradcam(models_dict, test_loader, class_names, save_dir=GRADCAM_COMPARE_DIR, num_images=6)

    # 5. 模型融合（简单平均集成）
    # 释放先前可能占用的显存，重新加载最佳模型以确保干净状态
    del models_dict, eff_model, max_model
    gc.collect()
    torch.cuda.empty_cache()

    eff_model = build_model('efficientnet', pretrained=False)
    eff_model.load_state_dict(torch.load('saved_models/EfficientNet_best.pth', map_location=DEVICE))
    max_model = build_model('maxvit', pretrained=False)
    max_model.load_state_dict(torch.load('saved_models/MaxViT_best.pth', map_location=DEVICE))

    print("生成模型融合预测结果...")
    eff_model.eval()
    max_model.eval()

    all_preds_fuse, all_labels_fuse = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            out1 = torch.softmax(eff_model(images), 1)
            out2 = torch.softmax(max_model(images), 1)
            fuse_out = 0.5 * out1 + 0.5 * out2          # 平均概率集成
            preds = fuse_out.argmax(1)
            all_preds_fuse.extend(preds.cpu().numpy())
            all_labels_fuse.extend(labels.numpy())

    report_fuse = classification_report(all_labels_fuse, all_preds_fuse, target_names=class_names)
    with open('ensemble_report.txt', 'w') as f:
        f.write(report_fuse)

    acc_fuse = accuracy_score(all_labels_fuse, all_preds_fuse)
    print(f'融合模型测试准确率: {acc_fuse:.4f}')
    cm_fuse = confusion_matrix(all_labels_fuse, all_preds_fuse)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_fuse, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Ensemble Confusion Matrix (Acc: {acc_fuse:.4f})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig('ensemble_confusion_matrix.png')
    plt.close()
    print("模型融合完成，报告已保存 ensemble_report.txt")