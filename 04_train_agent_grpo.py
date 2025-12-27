import re
import torch
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# 引入我们的环境 Client
from calculator_env.client import CalculatorEnv
from calculator_env.models import CalculatorAction

# ============================
# 配置
# ============================
SFT_MODEL_PATH = "sft-agent-0.6b"
OUTPUT_DIR = "grpo-agent-aligned"
ENV_URL = "http://localhost:8000" # 确保 server.app 正在运行

# ============================
# 自定义 Rollout 函数 (TRL的核心)
# ============================
def rollout_func(prompts, trainer, **kwargs):
    # 连接本地环境
    client = CalculatorEnv(base_url=ENV_URL)
    tokenizer = trainer.processing_class
    
    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    
    # 遍历 Batch 中的每一条 Prompt
    for i, prompt in enumerate(prompts):
        # 1. 解析题目参数，同步给环境
        # 假设 dataset 里的 prompt 包含了数字，我们用正则提取
        # 为了稳健，我们最好直接从 kwargs (dataset columns) 里拿，但在 rollout_func 签名里
        # TRL 目前主要传 prompts。我们尝试正则解析 prompt 文本。
        try:
            p1 = float(re.search(r"(\d+)% chance", prompt).group(1))
            vals = re.findall(r"\$(\d+)", prompt)
            v1, v2, sure = float(vals[0]), float(vals[1]), float(vals[2])
            client.set_problem(p1, v1, v2, sure)
            client.reset()
        except:
            pass # 解析失败就用环境默认值，防止 crash

        # 2. 多轮交互生成
        current_text = prompt
        generated_text = ""
        
        for _ in range(4): # 最多交互 4 轮
            inputs = tokenizer(current_text, return_tensors="pt").to(trainer.model.device)
            
            with torch.no_grad():
                # 生成模型回复
                outputs = trainer.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    stop_strings=["</tool>"], # 关键：遇到标签停止
                    tokenizer=tokenizer,
                    do_sample=True,
                    temperature=0.8
                )
            
            new_gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
            current_text += new_gen
            generated_text += new_gen
            
            # 处理工具调用
            if "</tool>" in new_gen:
                # 发送给环境
                action = CalculatorAction(message=new_gen)
                step_res = client.step(action)
                
                # 获取环境反馈
                obs = step_res.observation.feedback # <tool_output>...</tool_output>
                current_text += obs
                generated_text += obs
            
            elif "Final Decision" in new_gen:
                break # 结束
            else:
                break # 异常结束

        # 3. 整理结果
        # GRPO 需要 prompt_ids 和 completion_ids
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        c_ids = tokenizer(generated_text, add_special_tokens=False).input_ids
        
        all_prompt_ids.append(p_ids)
        all_completion_ids.append(c_ids)
        all_logprobs.append([0.0]*len(c_ids)) # 占位

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs
    }

# ============================
# 奖励函数
# ============================
def reward_outcome(completions, **kwargs):
    """结果奖励：是否做出了符合数学预期的决策"""
    rewards = []
    for i, text in enumerate(completions):
        # 获取真实参数
        p1, v1, v2 = kwargs['p1'][i], kwargs['v1'][i], kwargs['v2'][i]
        sure = kwargs['sure'][i]
        alpha, lam = kwargs['alpha'][i], kwargs['lambda'][i]
        
        def u(x): return x**alpha if x>=0 else -lam*((-x)**alpha)
        
        u_sure = u(sure)
        u_gamble = (p1/100 * u(v1)) + ((100-p1)/100 * u(v2))
        optimal = "accept" if u_sure > u_gamble else "reject"
        
        decision = "unknown"
        if "Final Decision: accept" in text: decision = "accept"
        if "Final Decision: reject" in text: decision = "reject"
        
        if decision == optimal:
            rewards.append(1.0)
        elif decision == "unknown":
            rewards.append(-1.0)
        else:
            rewards.append(-0.5)
    return rewards

def reward_format(completions, **kwargs):
    """过程奖励：是否正确使用了工具"""
    rewards = []
    for text in completions:
        score = 0.0
        if "<tool>" in text and "</tool>" in text:
            score += 0.2
        if "<tool_output>" in text: # 说明成功触发了环境反馈
            score += 0.3
        rewards.append(score)
    return rewards

# ============================
# 训练主流程
# ============================
dataset = load_from_disk("my_local_agent_data") # 加载本地生成的数据

model = AutoPeftModelForCausalLM.from_pretrained(SFT_MODEL_PATH, is_trainable=True)
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)

args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-6,
    num_generations=4,
    per_device_train_batch_size=2, # 显存小就调小
    gradient_accumulation_steps=4,
    max_completion_length=1024,
    use_vllm=False, # 使用自定义 rollout，关闭 vllm
    report_to="tensorboard"
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_outcome, reward_format],
    train_dataset=dataset,
    args=args,
    rollout_func=rollout_func, # 注入自定义 Agent 循环
)

print("🚀 Starting Agentic GRPO Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)