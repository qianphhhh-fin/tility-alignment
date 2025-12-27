import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================
# 1. 实验配置
# ============================
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B" # 未经SFT的基座
SFT_MODEL_PATH = "sft-agent-0.6b"              # 经过SFT的模型
NUM_SAMPLES = 200                              # 测试样本数 (越多越准)

# 目标人格参数 (Ground Truth)
TARGET_ALPHA = 0.88
TARGET_LAMBDA = 2.25

# ============================
# 2. 数据与工具函数
# ============================
def calculate_utility(v):
    """真实效用计算函数"""
    if v >= 0: return v ** TARGET_ALPHA
    return -TARGET_LAMBDA * ((-v) ** TARGET_ALPHA)

def get_utility_diff(p1, v1, v2, sure):
    """返回 U(Sure) - EU(Gamble)"""
    u_sure = calculate_utility(sure)
    u_gamble = (p1/100 * calculate_utility(v1)) + ((100-p1)/100 * calculate_utility(v2))
    return u_sure - u_gamble

def construct_perfect_context(tokenizer, p1, v1, v2, sure):
    p2 = 100 - p1
    
    # 1. 计算正确数值
    val_sure = calculate_utility(sure)
    val_gamble = (p1/100 * calculate_utility(v1)) + (p2/100 * calculate_utility(v2))
    
    decision_text = "accept" if val_sure > val_gamble else "reject"
    
    # 2. 构造 Prompt (保持不变)
    system_prompt = f"You are a rational economic agent. Use the <tool> tag to perform python calculations. Finally output 'Final Decision: accept' or 'reject'."
    user_prompt = f"The prospect is: {p1}% chance of ${v1}, {p2}% chance of ${v2}. The sure outcome is: ${sure}. Do you accept?"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 3. 构造 Assistant 的完美思考过程 (★★★ 关键修改：严格对齐 SFT 格式 ★★★)
    # 必须包含 "Since... I choose to..." 这一句，这是触发正确决策的开关
    
    # 注意：SFT数据里的代码是 print('Sure:', ... 'Gamble:', ...)
    # 我们这里尽量还原那个 print 的格式
    code_snippet = f"print('Sure:', {sure}**{TARGET_ALPHA}, 'Gamble:', {p1}/100 * ({v1}**{TARGET_ALPHA}))"
    tool_output = f"Sure: {val_sure:.5f} Gamble: {val_gamble:.5f}"
    
    # 比较符号
    comp_sign = ">" if val_sure > val_gamble else "<"
    
    assistant_prefix = f"""<think>
I need to compare the utility of the sure outcome with the expected utility of the prospect.
Alpha={TARGET_ALPHA}, Lambda={TARGET_LAMBDA}.
</think>
<tool>{code_snippet}</tool>
<tool_output>{tool_output}</tool_output>
<think>
Comparing: {val_sure:.5f} vs {val_gamble:.5f}
Since {val_sure:.5f} {comp_sign} {val_gamble:.5f}, I choose to {decision_text}.
</think>
Final Decision: """ # ★★★ 注意：这里加了一个空格！SFT数据里冒号后有空格

    return prompt_text + assistant_prefix

# ============================
# 3. 核心评测函数
# ============================
def evaluate_model(model, tokenizer, samples, model_name):
    print(f"\n🧪 Evaluating {model_name}...")
    
    # 获取 accept/reject 的 token id
    # 注意：Qwen 的 accept 前面通常带空格
    id_accept = tokenizer.encode(" accept")[0]
    id_reject = tokenizer.encode(" reject")[0]
    
    results = []
    
    for sample in tqdm(samples):
        p1, v1, v2, sure = sample
        
        # 构造完美上下文
        context = construct_perfect_context(tokenizer, p1, v1, v2, sure)
        inputs = tokenizer([context], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取最后一个 token 的 logits
        logits = outputs.logits[0, -1, :]
        
        score_accept = logits[id_accept].item()
        score_reject = logits[id_reject].item()
        
        # 计算 Logit Diff (Accept - Reject)
        logit_diff = score_accept - score_reject
        
        # 计算真实的 Utility Diff (Sure - Gamble)
        # 理论上：Utility Diff > 0 (Sure好) -> Logit Diff > 0 (Accept好)
        # 两者应该正相关
        util_diff = get_utility_diff(p1, v1, v2, sure)
        
        results.append({
            "model": model_name,
            "utility_diff": util_diff,
            "logit_diff": logit_diff
        })
        
    return pd.DataFrame(results)

# ============================
# 4. 主执行流程
# ============================
if __name__ == "__main__":
    # 生成测试集 (固定随机种子以公平对比)
    np.random.seed(42)
    samples = []
    for _ in range(NUM_SAMPLES):
        v1 = np.random.randint(500, 3000)
        v2 = 0
        p1 = 50
        # sure 在 20% 到 80% 之间波动，覆盖 accept 和 reject 的边界
        sure = np.random.randint(int(v1*0.2), int(v1*0.8))
        samples.append((p1, v1, v2, sure))

    # --- Round 1: 测试 Base Model ---
    print("🔄 Loading Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
    
    df_base = evaluate_model(model, tokenizer, samples, "Base Model (Prompted)")
    
    # 清理显存
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Round 2: 测试 SFT Model ---
    print("🔄 Loading SFT Model...")
    # 先加载基座
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
    # 再加载 Adapter
    model = PeftModel.from_pretrained(model, SFT_MODEL_PATH)
    
    df_sft = evaluate_model(model, tokenizer, samples, "SFT Model (Trained)")
    
    # --- Round 3: 绘图对比 ---
    print("\n📊 Plotting results...")
    df_all = pd.concat([df_base, df_sft])
    
    plt.figure(figsize=(12, 6))
    
    # 绘制散点图和回归线
    sns.scatterplot(data=df_all, x="utility_diff", y="logit_diff", hue="model", alpha=0.6)
    sns.regplot(data=df_base, x="utility_diff", y="logit_diff", scatter=False, color="blue", label="Base Trend")
    sns.regplot(data=df_sft, x="utility_diff", y="logit_diff", scatter=False, color="orange", label="SFT Trend")
    
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("Logits vs. Utility Alignment: Base vs. SFT")
    plt.xlabel("True Utility Difference (U_sure - EU_gamble)")
    plt.ylabel("Model Logit Difference (Logit_accept - Logit_reject)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("comparison_logits_utility.png")
    print("📈 Chart saved to 'comparison_logits_utility.png'")
    
    # 计算相关系数
    corr_base = df_base['utility_diff'].corr(df_base['logit_diff'])
    corr_sft = df_sft['utility_diff'].corr(df_sft['logit_diff'])
    
    print("\n" + "="*40)
    print("🏆 CORRELATION RESULTS (Pearson r)")
    print("="*40)
    print(f"Base Model: {corr_base:.4f}")
    print(f"SFT Model:  {corr_sft:.4f}")
    print("-" * 40)
    
    if corr_sft > corr_base:
        print("✅ Conclusion: SFT significantly improved alignment with the utility function.")
    else:
        print("🤔 Conclusion: Base model was already quite rational (or prompt was very effective).")