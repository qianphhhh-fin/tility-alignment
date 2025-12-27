import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================
# 配置
# ============================
# 使用中等规模模型 (Qwen2.5-7B-Instruct)
# 如果您有具体的 "Qwen/Qwen3-8B" 路径，请替换此处
MODEL_NAME = "Qwen/Qwen3-8B" 

# 目标人格参数 (用于生成 Agent 上下文中的正确计算结果)
TARGET_ALPHA = 0.88
TARGET_LAMBDA = 2.25

# 赌局设置：50% 赢 1000，50% 赢 0
V1 = 1000
V2 = 0
P1 = 50

# ============================
# 加载模型 (全精度，无量化)
# ============================
print(f"🔄 Loading {MODEL_NAME} in bfloat16 (No Quantization)...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, # 使用 bf16 全精度
        trust_remote_code=True
    ).eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Tip: Ensure you have enough VRAM (~15GB for 7B fp16).")
    exit()

# 获取 Token IDs
# Qwen 的 accept/reject token 通常前面带空格，视具体分词器而定
# 先打印检查一下
print("Checking Token IDs...")
t_acc = tokenizer.encode(" accept")
t_rej = tokenizer.encode(" reject")
print(f"' accept': {t_acc}")
print(f"' reject': {t_rej}")

id_accept = t_acc[0]
id_reject = t_rej[0]

# ============================
# 工具函数
# ============================
def calculate_utility(v):
    if v >= 0: return v ** TARGET_ALPHA
    return -TARGET_LAMBDA * ((-v) ** TARGET_ALPHA)

def get_prob_accept(context):
    inputs = tokenizer([context], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取最后一个 token 的 logits
    logits = outputs.logits[0, -1, :]
    score_accept = logits[id_accept].item()
    score_reject = logits[id_reject].item()
    
    # Softmax 计算 P(Accept)
    return np.exp(score_accept) / (np.exp(score_accept) + np.exp(score_reject))

# ============================
# 构造 Context (与 08 完全一致)
# ============================

# 1. Blind Context: 无工具，纯“裸考”
def build_blind_context(sure):
    sys = "You are a rational economic agent. Finally output 'Final Decision: accept' or 'reject'."
    user = f"The prospect is: {P1}% chance of ${V1}, {100-P1}% chance of ${V2}. The sure outcome is: ${sure}. Do you accept?"
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + "Final Decision:"

# 2. Agent Context: 注入工具计算结果 (Teacher Forcing)
def build_agent_context(sure):
    sys = "You are a rational economic agent. Use the <tool> tag to perform python calculations. Finally output 'Final Decision: accept' or 'reject'."
    user = f"The prospect is: {P1}% chance of ${V1}, {100-P1}% chance of ${V2}. The sure outcome is: ${sure}. Do you accept?"
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # --- 注入计算过程 ---
    u_sure = calculate_utility(sure)
    u_gamble = (P1/100 * calculate_utility(V1)) + ((100-P1)/100 * calculate_utility(V2))
    
    decision_text = "accept" if u_sure > u_gamble else "reject"
    comp_sign = ">" if u_sure > u_gamble else "<"
    
    # 即使模型没经过SFT，强大的Instruct模型通常也能理解这种思维链格式
    assistant_thought = f"""<think>
I need to compare the utility of the sure outcome with the expected utility of the prospect.
Alpha={TARGET_ALPHA}, Lambda={TARGET_LAMBDA}.
</think>
<tool>print('Sure:', {sure}**{TARGET_ALPHA}, 'Gamble:', {P1}/100 * ({V1}**{TARGET_ALPHA}))</tool>
<tool_output>Sure: {u_sure:.5f} Gamble: {u_gamble:.5f}</tool_output>
<think>
Comparing: {u_sure:.5f} vs {u_gamble:.5f}
Since {u_sure:.5f} {comp_sign} {u_gamble:.5f}, I choose to {decision_text}.
</think>
Final Decision:""" 

    return text + assistant_thought

# ============================
# 实验主循环
# ============================
# 理论切换点 CE
theory_eu = 0.5 * (1000 ** 0.88)
theory_ce = theory_eu ** (1/0.88) # ≈ 436.5

print(f"🧠 Theoretical Indifference Point (CE): ${theory_ce:.2f}")
print(f"🚀 Starting Psychometric Scan on {MODEL_NAME}...")

results = []
scan_range = range(0, 1050, 50)

for sure in tqdm(scan_range):
    # 1. Blind Test
    prob_blind = get_prob_accept(build_blind_context(sure))
    
    # 2. Agent Test (Context Injection)
    prob_agent = get_prob_accept(build_agent_context(sure))
    
    results.append({"Sure Amount": sure, "P(Accept)": prob_blind, "Condition": "Blind (No Tool)"})
    results.append({"Sure Amount": sure, "P(Accept)": prob_agent, "Condition": "Agent (With Tool)"})

# ============================
# 绘图
# ============================
df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Sure Amount", y="P(Accept)", hue="Condition", marker="o", linewidth=2.5)

# 参考线
plt.axvline(theory_ce, color='green', linestyle='--', label=f"Theoretical CE (${theory_ce:.0f})")
plt.axhline(0.5, color='red', linestyle=':', label="Decision Boundary")

plt.title(f"Psychometric Function: {MODEL_NAME} (Untrained)")
plt.xlabel("Sure Amount ($)")
plt.ylabel("Probability of Accepting Sure Option")
plt.legend()
plt.grid(True, alpha=0.3)

filename = "medium_model_psychometrics.png"
plt.savefig(filename)
print(f"\n📈 Plot saved to '{filename}'")