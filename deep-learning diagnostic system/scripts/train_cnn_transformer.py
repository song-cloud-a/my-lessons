import os
import copy
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# -------------------- 配置参数 --------------------
DATA_DIR = 'data/chest_xray'
MODEL_SAVE_PATH = 'best_cnn_transformer.pth'
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- 强数据增强 + CutMix/MixUp --------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# CutMix 和 MixUp 的辅助函数
def rand_bbox(size, lam):
    W, H = size[-2:]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_data(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size(0))
    labels_a, labels_b = labels, labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    return images, labels_a, labels_b, lam

def mixup_data(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size(0))
    labels_a, labels_b = labels, labels[rand_index]
    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, labels_a, labels_b, lam

# -------------------- 数据集加载 --------------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transforms)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class_names = train_dataset.classes
print(f"类别: {class_names}, 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

# -------------------- CNN-Transformer 模型定义 --------------------
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=2, cnn_backbone='resnet18', transformer_dim=512, depth=4, heads=8, mlp_dim=1024, dropout=0.1, stochastic_depth=0.1):
        super().__init__()
        # CNN 特征提取器 (移除最后的全连接和池化)
        if cnn_backbone == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # 输出 [B, 512, 7, 7]
            cnn_feat_dim = 512
        elif cnn_backbone == 'resnet50':
            backbone = models.resnet50(weights='IMAGENET1K_V1')
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            cnn_feat_dim = 2048
        else:
            raise ValueError("Unsupported backbone")

        # 将 CNN 特征映射到 transformer 维度
        self.input_proj = nn.Conv2d(cnn_feat_dim, transformer_dim, kernel_size=1)

        # 位置编码 (可学习的)
        seq_len = 7 * 7  # 49 个 patch
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, transformer_dim) * 0.02)

        # Transformer 编码器（使用 PyTorch 官方实现，但加入随机深度）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # 为每一层设置随机深度概率
        self.stochastic_depth_rate = stochastic_depth
        # 注意：PyTorch 的 TransformerEncoder 不支持逐层 drop path，这里简单使用 dropout 代替
        # 若要真正使用 DropPath，可借助 timm 库，但为了减少依赖，我们用 Dropout 在整个序列上
        # 我们在 forward 中手动实现逐层随机丢弃

        # 分类头
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Linear(transformer_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_cnn_features=False):
        # CNN 特征提取
        cnn_feat = self.cnn(x)  # [B, C, H, W]
        # 保存中间特征用于 Grad-CAM
        if return_cnn_features:
            return cnn_feat

        # 投影到 transformer 维度
        proj = self.input_proj(cnn_feat)  # [B, D, H, W]
        B, D, H, W = proj.shape
        proj = proj.flatten(2).transpose(1, 2)  # [B, N, D], N = H*W

        # 加上位置编码
        x = proj + self.pos_embed

        # 逐层 Transformer，并手动实现简单的随机深度
        for i, layer in enumerate(self.transformer.layers):
            # 随机深度：以一定概率跳过该层
            if self.training and self.stochastic_depth_rate > 0:
                if torch.rand(1).item() < self.stochastic_depth_rate:
                    continue  # 跳过该层
            x = layer(x)

        # 全局平均池化 (也可以取 [CLS] token，但这里用均值)
        x = x.mean(dim=1)
        x = self.norm(x)
        out = self.head(x)
        return out

# 实例化模型
model = CNNTransformer(
    num_classes=NUM_CLASSES,
    cnn_backbone='resnet18',   # 轻量以减少过拟合
    transformer_dim=256,       # 降低维度
    depth=4,                   # 较浅的 Transformer
    heads=8,
    mlp_dim=512,
    dropout=0.2,
    stochastic_depth=0.2
).to(DEVICE)

# -------------------- 损失函数（含 Label Smoothing）-------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# -------------------- 训练函数（集成 CutMix/MixUp）-------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    cutmix_prob = 0.5  # 一半概率使用 CutMix，另一半 MixUp

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # 随机选择 CutMix 或 MixUp
            if np.random.rand() < 0.5:
                # CutMix
                images, labels_a, labels_b, lam = cutmix_data(inputs, labels)
                optimizer.zero_grad()
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                # MixUp
                images, labels_a, labels_b, lam = mixup_data(inputs, labels)
                optimizer.zero_grad()
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

            loss.backward()
            optimizer.step()

            # 计算准确率 (基于混合标签近似)
            _, preds = torch.max(outputs, 1)
            correct = lam * (preds == labels_a).float().sum() + (1 - lam) * (preds == labels_b).float().sum()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += correct.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# -------------------- 测试评估 --------------------
def evaluate(model, dataloader):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = running_corrects.double() / len(dataloader.dataset)
    print(f'测试准确率: {accuracy:.4f}')
    return all_preds, all_labels, accuracy

# -------------------- Grad-CAM 可视化（基于 CNN 特征图）-------------------
def get_cam(model, img_tensor, class_idx=None):

    model.eval()
    # 获取 CNN 特征（需要梯度）
    img_tensor = img_tensor.to(DEVICE)
    img_tensor.requires_grad_()
    cnn_feat = model(img_tensor, return_cnn_features=True)  # [1, C, H, W]

    # 此时需要从 CNN 特征到最终输出的前向传播，以便反向传播梯度到 cnn_feat
    # 我们重新运行模型的其余部分，但保留 cnn_feat 的梯度连接
    # 简便方法：直接使用模型的完整 forward，但通过 hook 获取梯度
    # 这里采用更直接的方法：获取 CNN 特征后，手动执行后续步骤，确保梯度链不断
    model.zero_grad()
    cnn_feat.retain_grad()  # 让 cnn_feat 有梯度
    # 后续步骤：proj -> flatten -> pos_embed -> transformer -> mean -> norm -> head
    with torch.enable_grad():
        proj = model.input_proj(cnn_feat)
        B, D, H, W = proj.shape
        proj = proj.flatten(2).transpose(1, 2)
        x = proj + model.pos_embed
        for layer in model.transformer.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = model.norm(x)
        out = model.head(x)

        if class_idx is None:
            class_idx = torch.argmax(out, dim=1).item()
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1
        out.backward(gradient=one_hot, retain_graph=True)

    # 获取 CNN 特征的梯度
    gradients = cnn_feat.grad  # [1, C, H, W]
    # 全局平均池化梯度 -> 权重
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = torch.sum(weights * cnn_feat, dim=1, keepdim=True)  # [1, 1, H, W]
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.squeeze().cpu().detach().numpy()

def apply_heatmap(original_img, cam, alpha=0.5):
    cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = heatmap * alpha + original_img
    return np.clip(superimposed, 0, 255).astype(np.uint8)

def visualize_gradcam(model, dataloader, num_images=6):
    os.makedirs('gradcam_results', exist_ok=True)
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    for i in range(min(num_images, inputs.size(0))):
        img_tensor = inputs[i].unsqueeze(0)
        label = labels[i].item()
        cam = get_cam(model, img_tensor, class_idx=None)
        img_np = inputs[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        original_img = (img_np * 255).astype(np.uint8)
        superimposed = apply_heatmap(original_img, cam)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.imshow(original_img)
        plt.title(f'Original: {class_names[label]}')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(superimposed)
        plt.title(f'Grad-CAM (class: {class_names[label]})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'gradcam_results/sample_{i+1}.png')
        plt.close()
        print(f'样本 {i+1} 热力图已保存')

# -------------------- 主程序：训练 + 评估 + Grad-CAM --------------------
if __name__ == '__main__':
    model = train_model(model, criterion, optimizer, scheduler, EPOCHS)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'模型已保存至 {MODEL_SAVE_PATH}')
    preds, labels, _ = evaluate(model, test_loader)
    visualize_gradcam(model, test_loader, num_images=6)