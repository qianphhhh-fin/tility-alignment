import random
import pandas as pd
from datasets import Dataset, DatasetDict

# ============================
# 配置
# ============================
HF_USERNAME = "qianphhhh" # <--- 修改为你的用户名
REPO_ID = f"{HF_USERNAME}/risk-agent-dataset"
NUM_SAMPLES = 2000

# 目标人格 (Ground Truth)
TARGET_ALPHA = 0.88
TARGET_LAMBDA = 2.25

def calculate_utility(v):
    if v >= 0: return v ** TARGET_ALPHA
    return -TARGET_LAMBDA * ((-v) ** TARGET_ALPHA)

def generate_sample():
    # 1. 随机生成题目参数
    p1 = 50
    p2 = 50
    v1 = random.randint(500, 3000) # 金额大一点，必须用计算器
    v2 = 0
    sure = random.randint(int(v1*0.2), int(v1*0.6))
    
    # 2. 计算真实值 (Ground Truth)
    u_sure = calculate_utility(sure)
    u_gamble = (p1/100 * calculate_utility(v1)) + (p2/100 * calculate_utility(v2))
    decision = "accept" if u_sure > u_gamble else "reject"
    
    # 3. 构造 Prompt (输入)
    system_prompt = "You are a rational economic agent. Use the <tool> tag to perform python calculations. Finally output 'Final Decision: accept' or 'reject'."
    user_prompt = f"The prospect is: {p1}% chance of ${v1}, {p2}% chance of ${v2}. The sure outcome is: ${sure}. Do you accept?"
    
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    # 4. 构造 SFT 目标回复 (包含工具调用的完整轨迹)
    # 注意：我们教模型把计算分两步，或者一步写完皆可。这里教它写 Python print。
    
    code = f"print('Sure:', {sure}**{TARGET_ALPHA}, 'Gamble:', {p1}/100 * ({v1}**{TARGET_ALPHA}))"
    # 模拟工具输出 (保留5位小数)
    tool_output = f"Sure: {u_sure:.5f} Gamble: {u_gamble:.5f}"
    
    completion = f"""<think>
I need to compare the utility of the sure outcome with the expected utility of the prospect.
Alpha={TARGET_ALPHA}, Lambda={TARGET_LAMBDA}.
</think>
<tool>{code}</tool>
<tool_output>{tool_output}</tool_output>
<think>
Comparing: {u_sure:.5f} vs {u_gamble:.5f}
Since {u_sure:.5f} {" > " if u_sure > u_gamble else " < "} {u_gamble:.5f}, I choose to {decision}.
</think>
Final Decision: {decision}<|im_end|>"""

    return {
        "prompt": full_prompt,          # RL 用
        "completion": completion,       # SFT 用 (部分)
        "text": full_prompt + completion, # SFT 用 (完整)
        # 元数据 (RL Reward 用)
        "p1": float(p1), "v1": float(v1), "v2": float(v2), "sure": float(sure),
        "alpha": TARGET_ALPHA, "lambda": TARGET_LAMBDA
    }

# 生成数据
print(f"Generating {NUM_SAMPLES} samples...")
data = [generate_sample() for _ in range(NUM_SAMPLES)]
df = pd.DataFrame(data)

# 保存到本地
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("my_local_agent_data")
print("✅ Data generated and saved locally to 'my_local_agent_data'")

# (可选) 上传到 HF
# dataset.push_to_hub(REPO_ID)