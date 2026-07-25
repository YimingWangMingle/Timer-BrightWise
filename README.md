# Time Series Foundation Model

本项目用于实现一条可逐步扩展到 0.3B、1B 和 3B 的 Timer 风格 decoder-only
时间序列基础模型训练通路。目前里程碑是 CPU 可运行的微型端到端验证，不包含真实数据、
预训练权重或正式多卡训练。

## 当前能力

- 连续、无重叠 patch token；
- RoPE + PyTorch SDPA 因果自注意力；
- SiLU gated MLP 和 next-patch MSE；
- 只使用上下文统计量的归一化；
- 确定性合成时间序列；
- 自回归预测和任意长度裁剪；
- 模型、优化器和训练步数的原子检查点；
- 同一模型类通过 JSON 配置覆盖 tiny 与 0.3B 规模。

`third_party/` 只保存 Timer、Timer-XL 和 OpenLTM 的参考代码，运行时不会导入。

## 本地验证

在 PowerShell 中执行：

```powershell
cd D:\学习\TimeSeriesFoundationModel
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_smoke.py --steps 100
```

默认模型来自 `configs/model/tiny.json`，训练参数来自
`configs/training/smoke.json`。检查点写入 `outputs/smoke/final.pt`，该目录被 Git
忽略。

`configs/model/timer_300m.json` 约为 307.1M 参数，仅供解析与规模校验。本地不要实例化
该模型。

## 服务器边界

本地验证只证明前向、反向、因果性、归一化、生成和检查点链路正确。以下工作必须在服务器
完成：UTSD/LOTSA 下载与预处理、BF16、FlashAttention、NCCL/DDP 或 FSDP、H20 吞吐
测试、断点续训压力测试，以及 0.3B 正式训练。
