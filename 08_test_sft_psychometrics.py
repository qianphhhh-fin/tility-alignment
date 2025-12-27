import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# ============================
# 配置
# ============================
MODEL_PATH = "sft-agent-0.6b"
TARGET_ALPHA = 0.88
TARGET_LAMBDA = 2.25

# 赌局设置：50% 赢 1000，50% 赢 0
V1 = 1000
V2 = 0
P1 = 50

# ============================
# 加载模型
# ============================
print(f"🔄 Loading {MODEL_PATH}...")
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 获取 Token IDs
id_accept = tokenizer.encode(" accept")[0]
id_reject = tokenizer.encode(" reject")[0]

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
    logits = outputs.logits[0, -1, :]
    score_accept = logits[id_accept].item()
    score_reject = logits[id_reject].item()
    # Softmax
    return np.exp(score_accept) / (np.exp(score_accept) + np.exp(score_reject))

# ============================
# 构造两种 Context
# ============================

# 1. Blind Context: 只有题目，没有工具，没有思考
def build_blind_context(sure):
    sys = "You are a rational economic agent. Finally output 'Final Decision: accept' or 'reject'."
    user = f"The prospect is: {P1}% chance of ${V1}, {100-P1}% chance of ${V2}. The sure outcome is: ${sure}. Do you accept?"
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 强行引导到 Final Decision
    return text + "Final Decision:"

# 2. Agent Context (Tool-Augmented): 包含工具输出和推理逻辑
def build_agent_context(sure):
    sys = "You are a rational economic agent. Use the <tool> tag to perform python calculations. Finally output 'Final Decision: accept' or 'reject'."
    user = f"The prospect is: {P1}% chance of ${V1}, {100-P1}% chance of ${V2}. The sure outcome is: ${sure}. Do you accept?"
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # --- 注入完美思考过程 (Teacher Forcing) ---
    u_sure = calculate_utility(sure)
    u_gamble = (P1/100 * calculate_utility(V1)) + ((100-P1)/100 * calculate_utility(V2))
    
    decision_text = "accept" if u_sure > u_gamble else "reject"
    comp_sign = ">" if u_sure > u_gamble else "<"
    
    # 这里我们要完全模拟 SFT 数据的格式
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
Final Decision:""" # 停在这里，测 accept/reject 的概率

    return text + assistant_thought

# ============================
# 实验主循环
# ============================
results = []
# 扫描 Sure Amount 从 0 到 1000
# 理论切换点: 1000^0.88 * 0.5 = 436^0.88 ≈ 218 -> Sure^0.88 = 218 -> Sure = 218^(1/0.88) ≈ 436
# (因为 v2=0，且 p=0.5，其实切换点就在 1000*0.5 * (概率权重) 附近，线性近似下接近期望值，但受 alpha 影响)
# 准确计算理论切换点：
theory_eu = 0.5 * (1000 ** 0.88)
theory_ce = theory_eu ** (1/0.88) # 应该是 436.5

print(f"🧠 Theoretical Indifference Point (CE): ${theory_ce:.2f}")
print("🚀 Starting Psychometric Scan...")

scan_range = range(0, 1050, 50)

for sure in tqdm(scan_range):
    # 1. Test Blind SFT
    prob_blind = get_prob_accept(build_blind_context(sure))
    
    # 2. Test Agent SFT (With Tool Info)
    prob_agent = get_prob_accept(build_agent_context(sure))
    
    results.append({"Sure Amount": sure, "P(Accept)": prob_blind, "Condition": "Blind SFT (No Tool)"})
    results.append({"Sure Amount": sure, "P(Accept)": prob_agent, "Condition": "Agent SFT (With Tool)"})

# ============================
# 绘图
# ============================
df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Sure Amount", y="P(Accept)", hue="Condition", marker="o", linewidth=2.5)

# 参考线
plt.axvline(theory_ce, color='green', linestyle='--', label=f"Theoretical CE (${theory_ce:.0f})")
plt.axhline(0.5, color='red', linestyle=':', label="Decision Boundary")

plt.title("Psychometric Function: Blind vs. Agentic SFT Model")
plt.xlabel("Sure Amount ($)")
plt.ylabel("Probability of Accepting Sure Option")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("sft_agent_psychometrics.png")
print("\n📈 Plot saved to 'sft_agent_psychometrics.png'")