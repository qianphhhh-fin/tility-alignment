import torch
import re
import json
import random
import numpy as np
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# 引入 OpenEnv 客户端
from calculator_env.client import CalculatorEnv
from calculator_env.models import CalculatorAction

# ============================
# 配置
# ============================
MODEL_PATH = "sft-agent-0.6b"
ENV_URL = "http://localhost:8000"
TEST_SAMPLES = 100  # 测试样本数量
LOG_FILE = "batch_test_results.jsonl"

# 目标人格参数 (用于计算标准答案 Ground Truth)
TARGET_ALPHA = 0.88
TARGET_LAMBDA = 2.25

# ============================
# 辅助函数
# ============================
def calculate_ground_truth(p1, v1, v2, sure):
    """计算理性决策的标准答案"""
    def u(x):
        if x >= 0: return x ** TARGET_ALPHA
        return -TARGET_LAMBDA * ((-x) ** TARGET_ALPHA)
    
    u_sure = u(sure)
    u_gamble = (p1/100 * u(v1)) + ((100-p1)/100 * u(v2))
    
    return "accept" if u_sure > u_gamble else "reject"

def generate_random_problem():
    """生成随机测试题"""
    p1 = 50
    v1 = random.randint(500, 3000)
    v2 = 0
    sure = random.randint(int(v1*0.2), int(v1*0.6)) # 覆盖 accept 和 reject 的边界
    return p1, v1, v2, sure

# ============================
# 初始化
# ============================
print(f"🔄 Loading Model: {MODEL_PATH}...")
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

try:
    print(f"🌐 Connecting to Environment at {ENV_URL}...")
    client = CalculatorEnv(base_url=ENV_URL)
    client.reset()
    print("✅ Environment Connected!")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    print("Please run 'python -m calculator_env.server.app' first.")
    exit()

# ============================
# 单次推理逻辑 (Agent Loop)
# ============================
def run_single_test(p1, v1, v2, sure):
    p2 = 100 - p1
    
    # 1. 设置题目
    client.set_problem(p1, v1, v2, sure)
    client.reset()
    
    # 2. 构造 Prompt
    system_prompt = "You are a rational economic agent. Use the <tool> tag to perform python calculations. Finally output 'Final Decision: accept' or 'reject'."
    user_prompt = f"The prospect is: {p1}% chance of ${v1}, {p2}% chance of ${v2}. The sure outcome is: ${sure}. Do you accept?"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    trajectory = text # 记录完整对话历史
    tool_used = False
    tool_error = False
    decision = "unknown"
    
    # 3. 多轮交互循环
    for _ in range(4): # 最多允许交互 4 轮
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                stop_strings=["</tool>"], # 只在工具调用结束时停，让它自己输出 Final Decision
                tokenizer=tokenizer,
                do_sample=False # Greedy Decoding 测能力边界
            )
            
        new_content = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        text += new_content
        trajectory += new_content
        
        # A. 检查是否调用工具
        if "</tool>" in new_content:
            tool_used = True
            action = CalculatorAction(message=new_content)
            try:
                # 调用 OpenEnv
                step_result = client.step(action)
                feedback = step_result.observation.feedback
                
                # 检查是否报错
                if "Error" in feedback:
                    tool_error = True
                
                # 拼接到上下文
                text += feedback
                trajectory += feedback
                
            except Exception:
                tool_error = True
                break
        
        # B. 检查是否做出决策
        # 注意：需要处理可能的额外字符，用正则提取
        if "Final Decision" in new_content:
            match = re.search(r"Final Decision:\s*(accept|reject)", new_content, re.IGNORECASE)
            if match:
                decision = match.group(1).lower()
            break
            
    return decision, tool_used, tool_error, trajectory

# ============================
# 批量测试主流程
# ============================
print(f"\n🚀 Starting Batch Evaluation ({TEST_SAMPLES} samples)...")
print(f"📝 Logs will be saved to: {LOG_FILE}")

stats = {
    "total": 0,
    "correct": 0,
    "tool_used_correctly": 0,
    "tool_errors": 0,
    "format_errors": 0
}

# 清空旧日志
open(LOG_FILE, 'w').close()

for i in tqdm(range(TEST_SAMPLES)):
    # 1. 生成题目
    p1, v1, v2, sure = generate_random_problem()
    ground_truth = calculate_ground_truth(p1, v1, v2, sure)
    
    # 2. 运行 Agent
    model_decision, tool_used, tool_error, trace = run_single_test(p1, v1, v2, sure)
    
    # 3. 统计
    stats["total"] += 1
    is_correct = (model_decision == ground_truth)
    
    if is_correct:
        stats["correct"] += 1
    
    if tool_used and not tool_error:
        stats["tool_used_correctly"] += 1
        
    if tool_error:
        stats["tool_errors"] += 1
        
    if model_decision == "unknown":
        stats["format_errors"] += 1
        
    # 4. 实时记录日志
    log_entry = {
        "id": i,
        "problem": {"p1": p1, "v1": v1, "v2": v2, "sure": sure},
        "ground_truth": ground_truth,
        "model_decision": model_decision,
        "is_correct": is_correct,
        "tool_used": tool_used,
        "tool_error": tool_error,
        "full_trace": trace
    }
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ============================
# 最终报告
# ============================
print("\n" + "="*40)
print("📊 EVALUATION REPORT")
print("="*40)
print(f"Total Samples:      {stats['total']}")
print(f"✅ Accuracy:         {stats['correct'] / stats['total'] * 100:.2f}%")
print(f"🛠️ Tool Usage Rate:  {stats['tool_used_correctly'] / stats['total'] * 100:.2f}% (Valid Calls)")
print(f"⚠️ Tool Errors:      {stats['tool_errors']}")
print(f"❌ Format Errors:    {stats['format_errors']} (No Decision)")
print("="*40)

if stats['correct'] / stats['total'] > 0.9:
    print("🎉 CONGRATULATIONS! Model is ready. No RL needed.")
else:
    print("💪 Good start, but needs RL (Step 04) to improve robustness.")