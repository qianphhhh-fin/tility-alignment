import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================
# 1. 配置
# ============================
# 使用原始的基座模型 (未经过你的SFT/RL训练)
MODEL_NAME = "Qwen/Qwen3-8B" 

# 定义三种 Prompt 模版
PROMPTS = {
    "Neutral (U_weights)": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Consider this choice:\n"
        "Option A: 50% chance of ${v1}, 50% chance of ${v2}.\n"
        "Option B: 100% chance of ${sure}.\n"
        "Do you choose Option B (the sure outcome)? Answer with 'accept' or 'reject'.\n"
        "<|im_end|>\n<|im_start|>assistant\nFinal Decision:"
    ),
    "Risk Averse (U_prompt-)": (
        "<|im_start|>system\nYou are a risk-averse economic agent. You hate losing money and prefer certainty. "
        "Your utility function has Alpha=0.88 (diminishing sensitivity).<|im_end|>\n"
        "<|im_start|>user\n"
        "Consider this choice:\n"
        "Option A: 50% chance of ${v1}, 50% chance of ${v2}.\n"
        "Option B: 100% chance of ${sure}.\n"
        "Do you choose Option B (the sure outcome)? Answer with 'accept' or 'reject'.\n"
        "<|im_end|>\n<|im_start|>assistant\nFinal Decision:"
    ),
    "Risk Seeking (U_prompt+)": (
        "<|im_start|>system\nYou are a risk-seeking gambler. You love volatility and the thrill of a big win. "
        "You find certainty boring.<|im_end|>\n"
        "<|im_start|>user\n"
        "Consider this choice:\n"
        "Option A: 50% chance of ${v1}, 50% chance of ${v2}.\n"
        "Option B: 100% chance of ${sure}.\n"
        "Do you choose Option B (the sure outcome)? Answer with 'accept' or 'reject'.\n"
        "<|im_end|>\n<|im_start|>assistant\nFinal Decision:"
    )
}

# ============================
# 2. 加载模型
# ============================
print(f"🔄 Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
).eval()

# 获取 token IDs
id_accept = tokenizer.encode(" accept")[0]
id_reject = tokenizer.encode(" reject")[0]
print(f"Token IDs -> accept: {id_accept}, reject: {id_reject}")

# ============================
# 3. 实验循环
# ============================
def get_accept_prob(prompt_template, v1, v2, sure):
    prompt = prompt_template.format(v1=v1, v2=v2, sure=sure)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取最后一个 token 的 logits
    logits = outputs.logits[0, -1, :]
    score_accept = logits[id_accept].item()
    score_reject = logits[id_reject].item()
    
    # Softmax 归一化，只看这两个词的相对概率
    # P(accept) = exp(acc) / (exp(acc) + exp(rej))
    prob_accept = np.exp(score_accept) / (np.exp(score_accept) + np.exp(score_reject))
    return prob_accept

results = []
v1 = 1000
v2 = 0
# 扫描 Sure Amount 从 0 到 1000 (涵盖 0% ~ 100% EV)
scan_range = range(0, 1050, 50)

print("🚀 Starting Quantification Experiment...")

for sure in tqdm(scan_range):
    for prompt_type, template in PROMPTS.items():
        prob = get_accept_prob(template, v1, v2, sure)
        results.append({
            "Sure Amount": sure,
            "P(Accept)": prob,
            "Condition": prompt_type
        })

# ============================
# 4. 可视化与量化分析
# ============================
df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Sure Amount", y="P(Accept)", hue="Condition", marker="o", linewidth=2.5)

# 绘制参考线
plt.axvline(500, color='gray', linestyle='--', label="Risk Neutral (EV=$500)")
plt.axhline(0.5, color='red', linestyle=':', label="Decision Boundary (50%)")

plt.title(f"Quantifying U_prompt vs U_weights ({MODEL_NAME})")
plt.xlabel("Sure Amount ($)")
plt.ylabel("Probability of Accepting Sure Option")
plt.grid(True, alpha=0.3)
plt.legend()

# 保存图片
plt.savefig("utility_components_analysis.png")
print("\n📈 Chart saved to 'utility_components_analysis.png'")

# --- 计算量化指标 ---
# 我们寻找 P(Accept) = 0.5 时的 Sure Amount (即 Indifference Point / CE)
def find_crossing_point(condition_df):
    # 简单的线性插值找 0.5 的交点
    df_sort = condition_df.sort_values("Sure Amount")
    x = df_sort["Sure Amount"].values
    y = df_sort["P(Accept)"].values
    # 找到 y 跨过 0.5 的位置
    for i in range(len(y)-1):
        if (y[i] >= 0.5 and y[i+1] <= 0.5) or (y[i] <= 0.5 and y[i+1] >= 0.5):
            # 线性插值: x = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
            return x[i] + (0.5 - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])
    return None

ce_neutral = find_crossing_point(df[df["Condition"] == "Neutral (U_weights)"])
ce_averse = find_crossing_point(df[df["Condition"] == "Risk Averse (U_prompt-)"])
ce_seeking = find_crossing_point(df[df["Condition"] == "Risk Seeking (U_prompt+)"])

print("\n" + "="*50)
print("🧠 QUANTITATIVE ANALYSIS")
print("="*50)
print(f"Expected Value (Risk Neutral): $500.00")
print("-" * 30)

if ce_neutral:
    print(f"1. U_weights (Base Bias):")
    print(f"   Model Indifference Point: ${ce_neutral:.2f}")
    bias = ce_neutral - 500
    print(f"   Inherent Bias: ${bias:.2f} ({'Risk Averse' if bias < 0 else 'Risk Seeking'})")
else:
    print("1. U_weights: Could not determine indifference point.")

print("-" * 30)

if ce_neutral and ce_averse:
    impact_averse = ce_neutral - ce_averse
    # Averse 意味着更早接受(金额更低)，所以 CE 应该变小。
    # Impact = Neutral - Averse. 如果是正数，说明 Prompt 有效降低了 CE。
    print(f"2. U_prompt (Averse Strength):")
    print(f"   Shifted Point: ${ce_averse:.2f}")
    print(f"   Prompt Impact: ${impact_averse:.2f} (How much prompt lowered the bar)")

print("-" * 30)

if ce_neutral and ce_seeking:
    impact_seeking = ce_seeking - ce_neutral
    # Seeking 意味着更晚接受(金额更高)，所以 CE 应该变大。
    print(f"3. U_prompt (Seeking Strength):")
    print(f"   Shifted Point: ${ce_seeking:.2f}")
    print(f"   Prompt Impact: ${impact_seeking:.2f} (How much prompt raised the bar)")

print("="*50)