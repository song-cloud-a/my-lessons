```markdown
# 基于深度学习的胸部 X 光肺炎识别系统

本项目使用 PyTorch 构建胸部 X 光肺炎二分类模型，实现对 `NORMAL`（正常）和 `PNEUMONIA`（肺炎）图像的自动识别。

项目包含 EfficientNet-B0、MaxViT-Tiny、CNN、CNN-Transformer 和双模型概率融合实验，并使用 Grad-CAM 展示模型在预测过程中重点关注的图像区域。

> **免责声明：本项目仅用于机器学习研究与教学，不能替代医生诊断，也不应直接用于临床医疗场景。**

![Grad-CAM 对比结果](assets/results/gradcam_comparison/sample_1.png)

## 项目功能

- 胸部 X 光图像二分类
- EfficientNet-B0 模型训练与评估
- MaxViT-Tiny 模型训练与评估
- CNN 基线模型实验
- CNN-Transformer 混合模型实验
- EfficientNet 与 MaxViT 概率融合
- Grad-CAM 模型可解释性分析
- Gradio 图形化预测界面
- 混淆矩阵、ROC 曲线和训练曲线生成
- GitHub Actions 自动代码检查

## 实验结果

测试集共包含 624 张胸部 X 光图像：

- NORMAL：234 张
- PNEUMONIA：390 张

| 模型 | Accuracy | NORMAL F1 | PNEUMONIA F1 |
| --- | ---: | ---: | ---: |
| EfficientNet-B0 | 0.93 | 0.90 | 0.95 |
| MaxViT-Tiny | 0.96 | 0.95 | 0.97 |
| 双模型概率融合 | 0.96 | 0.95 | 0.97 |

详细评估报告位于：

```text
assets/reports/
```

训练曲线、混淆矩阵和 ROC 曲线位于：

```text
assets/results/metrics/
```

以上结果仅适用于当前数据集和数据划分，不代表模型在真实临床环境中的性能。

## 项目结构

```text
.
├── app.py
├── scripts/
│   ├── train_cnn.py
│   ├── train_cnn_transformer.py
│   └── train_ensemble.py
├── assets/
│   ├── reports/
│   │   ├── EfficientNet_report.txt
│   │   ├── MaxViT_report.txt
│   │   └── ensemble_report.txt
│   └── results/
│       ├── metrics/
│       ├── gradcam/
│       └── gradcam_comparison/
├── tests/
├── .github/
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## 环境要求

推荐使用：

- Python 3.10 或 Python 3.11
- PyTorch 2.2+
- Windows、Linux 或 macOS
- 推荐使用支持 CUDA 的 NVIDIA GPU 进行训练

## 安装方法

克隆项目：

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

Linux 或 macOS 激活环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 数据集目录

训练代码使用 PyTorch `ImageFolder` 加载数据，目录结构应为：

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

数据集体积较大，因此没有上传到 GitHub。

使用者需要自行准备数据集，并确保拥有对应的数据使用权。涉及真实患者的数据应完成隐私保护、去标识化和必要的合规审查。

## 模型训练

训练 EfficientNet-B0、MaxViT-Tiny 并进行融合评估：

```bash
python scripts/train_ensemble.py
```

运行 CNN 基线实验：

```bash
python scripts/train_cnn.py
```

运行 CNN-Transformer 实验：

```bash
python scripts/train_cnn_transformer.py
```

训练完成后的模型权重默认保存在：

```text
saved_models/
├── EfficientNet_best.pth
└── MaxViT_best.pth
```

## 启动预测界面

准备好模型权重后运行：

```bash
python app.py
```

浏览器访问：

```text
http://127.0.0.1:7860
```

在页面中上传胸部 X 光图像并选择模型，即可查看：

- 模型预测结果
- NORMAL 和 PNEUMONIA 分类概率
- Grad-CAM 模型关注区域

## 模型权重

由于模型权重文件较大，项目没有将 `.pth` 和 `.safetensors` 文件直接上传到 GitHub 仓库。

请将模型权重放入：

```text
saved_models/
```

也可以使用环境变量指定模型权重目录。

Windows PowerShell：

```powershell
$env:PNEUMONIA_MODEL_DIR="D:\models\pneumonia"
python app.py
```

模型权重可以通过以下方式发布：

- GitHub Release
- Git LFS
- Hugging Face Hub

## Grad-CAM 可解释性分析

Grad-CAM 用于展示模型在进行分类时重点关注的区域。

单模型 Grad-CAM 结果位于：

```text
assets/results/gradcam/
```

多模型对比结果位于：

```text
assets/results/gradcam_comparison/
```

需要注意，Grad-CAM 只能辅助理解模型行为，不能作为医学诊断依据，也不能证明模型预测与某个图像区域之间存在因果关系。

## 开发与测试

安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt
```

运行代码检查：

```bash
ruff check app.py tests
```

运行自动测试：

```bash
python -m pytest
```

## 项目局限性

- 当前数据类别分布不完全平衡。
- 验证集规模较小，评估结果可能存在一定方差。
- 模型可能学习到数据集中的设备、标记或图像处理差异。
- 公开数据集与真实医疗环境之间可能存在域偏移。
- 当前结果尚未经过独立外部数据集验证。
- 模型没有经过临床审核、校准或医疗器械认证。
- 本项目不能用于实际医疗决策。

## 后续改进方向

- 扩大并重新划分验证集
- 增加独立外部测试集
- 加入类别平衡采样策略
- 增加模型置信度校准
- 支持批量图像预测
- 增加更多模型对比实验
- 提供 Hugging Face 模型权重
- 使用 Docker 简化部署
- 增加完整训练参数配置文件

## 贡献

欢迎通过 Issue 或 Pull Request 提交建议和改进。

提交代码前建议运行：

```bash
ruff check app.py tests
python -m pytest
```

请勿提交患者隐私数据、原始数据集、模型权重或其他敏感信息。

## License

本项目代码使用 [MIT License](LICENSE)。

数据集与模型权重可能具有独立的许可证和使用限制，使用前请确认对应条款。
```
