
---

# 🧠 Utility Alignment: LLM Economic Rationality Experiment
# 经济效用对齐：基于前景理论的 LLM Agent 实验

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

本项目旨在探索如何通过 **监督微调 (SFT)** 和 **强化学习 (GRPO/RL)**，将大型语言模型 (LLM) 的决策行为与特定的经济学效用函数（[前景理论 Prospect Theory](https://en.wikipedia.org/wiki/Prospect_theory)）进行对齐。

通过让模型学习使用 Python 工具计算效用值，并结合思维链 (CoT)，我们将一个基础的小型模型 (`Qwen/Qwen3-0.6B`) 训练成了一个具备特定风险偏好（$\alpha=0.88, \lambda=2.25$）的理性经济 Agent。

## 🎯 核心目标

1.  **人格对齐 (Alignment):** 强制模型遵循特定的人类经济偏好（如损失厌恶、边际效用递减），而非仅仅是风险中性 (Expected Value) 或预训练时的随机偏好。
2.  **工具增强 (Tool-Use):** 训练模型在决策前主动调用 Python 计算器，确保数值计算的准确性，避免大模型的算术幻觉。
3.  **心理测量 (Psychometrics):** 使用心理物理学方法（Psychometric Functions）量化评估模型的风险偏好曲线。

## 🛠️ 技术栈

*   **Model:** Qwen/Qwen3-0.6B (SFT/RL), Qwen/Qwen3-8B (Baseline Analysis)
*   **Training:** LoRA (PEFT), SFTTrainer, GRPOTrainer (TRL)
*   **Environment:** 自定义 `calculator_env` (Server-Client 架构)
*   **Analysis:** Pandas, Seaborn, Matplotlib

## 📦 安装与准备

### 1. 环境依赖
请确保安装了以下 Python 库：
```bash
pip install torch transformers peft datasets trl pandas matplotlib seaborn tqdm numpy
```

### 2. 启动环境服务器
本项目依赖一个外部计算环境来处理 Agent 的工具调用。请确保 `calculator_env` 服务已启动（代码中默认为 `http://localhost:8000`）：
```bash
# 假设 calculator_env 在你的 python path 中
python -m calculator_env.server.app
```

## 🚀 工作流程 (Pipeline)

### Step 1: 数据生成
生成符合前景理论 ($\alpha=0.88, \lambda=2.25$) 的合成数据集。数据包含 `<think>` 思考过程和 `<tool>` 工具调用轨迹。
```bash
python 01_create_agent_sft_data.py
```
*   输出: `my_local_agent_data` (HuggingFace Dataset 格式)

### Step 2: 监督微调 (SFT)
让模型学会两件事：1. 理解经济学术语；2. 学会正确的工具调用格式。
```bash
python 02_train_sft.py
```
*   模型: `Qwen/Qwen3-0.6B` + LoRA
*   输出: `sft-agent-0.6b`

### Step 3: SFT 模型评估
测试 SFT 后的模型在交互式环境中的表现，检查工具调用成功率和决策准确率。
```bash
python 03_batch_test_sft.py
# 或进行交互式单测
python 03_test_sft_interactive.py
```
*   日志: `batch_test_results.jsonl`

### Step 4: 强化学习对齐 (GRPO)
使用 **Group Relative Policy Optimization (GRPO)** 进一步优化模型。
*   **Reward Model:** 基于决策是否符合数学最优解 (Outcome Reward) + 格式正确性 (Format Reward)。
*   **Rollout:** 在本地环境中采样多条轨迹。
```bash
python 04_train_agent_grpo.py
```

## 📊 分析与可视化

本项目包含丰富的分析脚本，用于量化模型的对齐程度。

| 脚本 | 描述 |
| :--- | :--- |
| `06_compare_logits_utility.py` | **Logits 分析**: 比较模型输出 token (`accept`/`reject`) 的 Logits 差值与真实效用差值 ($U_{sure} - EU_{gamble}$) 的相关性。 |
| `07_quantify_prompt_vs_weights.py` | **Prompt 效应量化**: 分析 Prompt 中设定的角色（如“风险厌恶” vs “赌徒”）对模型决策阈值 (Indifference Point) 的定量影响。 |
| `08_test_sft_psychometrics.py` | **SFT 心理测量**: 绘制 SFT 模型的心理测量曲线 (S-Curve)，对比由无工具 ("Blind") 和有工具 ("Agentic") 状态下的决策概率。 |
| `09_test_medium_model_psychometrics.py` | **基座模型分析**: 在未微调的中等模型 (Qwen-8B) 上进行同样的心理测量分析作为对照组。 |

### 结果示例 (Plots)
运行上述脚本将生成以下图表：
*   `comparison_logits_utility.png`: Logits 与真实效用的线性回归图。
*   `utility_components_analysis.png`: 不同 Prompt 下模型的确定性效应 (Certainty Effect) 偏移。
*   `sft_agent_psychometrics.png`: 概率接受曲线，展示模型是否在理论无差异点 (Theoretical CE) 附近发生翻转。

## 📂 目录结构

```text
tility-alignment-main/
├── 01_create_agent_sft_data.py       # 数据生成
├── 02_train_sft.py                   # SFT 训练脚本
├── 03_batch_test_sft.py              # 批量测试与评估
├── 03_test_sft_interactive.py        # 单例交互测试
├── 04_train_agent_grpo.py            # GRPO 强化学习训练
├── 06_compare_logits_utility.py      # Logits vs Utility 分析
├── 07_quantify_prompt_vs_weights.py  # Prompt 敏感度分析
├── 08_test_sft_psychometrics.py      # SFT 模型心理测量曲线
├── 09_test_medium_model_psychometrics.py # 基座模型对照分析
├── batch_test_results.jsonl          # 测试日志
└── my_local_agent_data/              # 本地数据集目录
```

## 📝 备注

*   **Security:** `repomix` 处理的文件中禁用了安全检查，请确保在受控环境中运行代码，尤其是涉及 `exec` 或工具调用的部分。
*   **Data:** 当前数据集基于合成生成的二元选择题 (Sure thing vs Gamble)。
*   **Hardware:** 0.6B 模型训练可在单张消费级显卡 (如 RTX 3060/4090) 上完成。GRPO 训练建议使用较大显存。

---
*Created by [Your Username]*