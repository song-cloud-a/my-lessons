# Chest X-ray Pneumonia Classification

基于 PyTorch 的胸部 X 光肺炎二分类项目，包含 EfficientNet-B0、MaxViT-Tiny、概率融合、Grad-CAM 可解释性分析和 Gradio 交互演示。

> [!IMPORTANT]
> 本项目仅用于机器学习研究与教学，输出结果不能替代放射科医生或其他医疗专业人员的诊断。

![Grad-CAM comparison](assets/results/gradcam_comparison/sample_1.png)

## 项目亮点

- 对比卷积网络 EfficientNet-B0 与视觉 Transformer 架构 MaxViT-Tiny
- 使用迁移学习、MixUp/CutMix、EMA 等训练策略
- 提供单模型和双模型概率融合评估
- 通过 Grad-CAM 展示模型关注区域
- 提供开箱即用的 Gradio Web 界面
- 配置 GitHub Actions、Ruff 和轻量级自动测试

## 实验结果

以下结果来自仓库内保存的测试报告，测试集共 624 张图像（NORMAL 234 张，PNEUMONIA 390 张）。

| 模型 | Accuracy | NORMAL F1 | PNEUMONIA F1 |
| --- | ---: | ---: | ---: |
| EfficientNet-B0 | 0.93 | 0.90 | 0.95 |
| MaxViT-Tiny | 0.96 | 0.95 | 0.97 |
| 概率融合 | 0.96 | 0.95 | 0.97 |

详细结果可查看 `assets/reports/`、混淆矩阵、ROC 曲线和训练曲线。指标仅代表当前数据划分，不表示临床性能。

## 快速开始

建议使用 Python 3.10 或 3.11。

```bash
git clone https://github.com/<your-name>/<your-repository>.git
cd <your-repository>
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。

## 模型权重

Web 演示默认读取以下文件：

```text
saved_models/
├── EfficientNet_best.pth
└── MaxViT_best.pth
```

模型权重体积较大，已被 `.gitignore` 排除。发布时建议上传至 GitHub Release、Hugging Face Hub，或使用 Git LFS，并在此处补充公开下载地址。也可通过环境变量指定权重目录：

```powershell
$env:PNEUMONIA_MODEL_DIR="D:\models\pneumonia"
python app.py
```

## 数据集

代码期望使用如下 ImageFolder 目录结构：

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

数据集不会提交到 GitHub。请确保你拥有数据使用权，并在公开项目时补充数据来源、许可证和下载说明。涉及真实患者的数据必须先完成合规审查与去标识化。

## 使用方式

启动交互演示：

```bash
python app.py
```

运行完整双模型训练与评估：

```bash
python scripts/train_ensemble.py
```

其他实验脚本：

- `scripts/train_cnn.py`：CNN 基线实验
- `scripts/train_cnn_transformer.py`：CNN-Transformer 实验
- `scripts/train_ensemble.py`：EfficientNet、MaxViT、融合评估及 Grad-CAM

当前训练脚本的参数定义在脚本顶部。首次运行预训练模型时可能需要联网下载 ImageNet 权重。

## 项目结构

```text
.
├── app.py                         # Gradio 推理与 Grad-CAM 演示
├── scripts/                       # 训练与评估脚本
│   ├── train_ensemble.py          # 双模型训练、评估与融合
│   ├── train_cnn.py               # CNN 基线
│   └── train_cnn_transformer.py   # CNN-Transformer 实验
├── assets/
│   ├── reports/                   # 模型指标报告
│   └── results/
│       ├── metrics/               # 曲线、ROC 与混淆矩阵
│       ├── gradcam/               # Grad-CAM 示例
│       └── gradcam_comparison/    # 多模型可视化对比
├── tests/                          # 快速仓库检查
├── requirements.txt
└── pyproject.toml
```

## 开发与验证

```bash
python -m pip install -r requirements-dev.txt
ruff check app.py tests
python -m pytest
```

欢迎通过 Issue 或 Pull Request 改进项目，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 局限性

- 数据类别分布不平衡，且当前验证集很小，结果可能存在较大方差。
- 公开数据集与真实临床场景可能存在显著域偏移。
- Grad-CAM 只能辅助理解模型关注区域，不能证明因果关系或诊断依据。
- 在独立外部数据集验证、校准与临床评审完成前，不应部署到医疗工作流。

## License

代码使用 [MIT License](LICENSE)。数据集和模型权重可能适用各自独立的许可条款。
