# 贡献指南

感谢参与改进。请先创建分支，并保持一次提交只解决一个明确问题。

## 本地检查

```bash
python -m pip install -r requirements-dev.txt
ruff check app.py tests
python -m pytest
```

提交 Pull Request 时，请说明修改动机、验证方式，以及模型指标变化（如有）。请勿提交患者数据、原始数据集、模型权重或其他敏感信息。
