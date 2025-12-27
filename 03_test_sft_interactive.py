import torch
import re
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# 引入 OpenEnv 客户端
from calculator_env.client import CalculatorEnv
from calculator_env.models import CalculatorAction

# ============================
# 配置
# ============================
MODEL_PATH = "sft-agent-0.6b"
ENV_URL = "http://localhost:8000" # 确保你的服务器在运行

# ============================
# 加载模型
# ============================
print(f"🔄 Loading Model: {MODEL_PATH}...")
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ============================
# 初始化环境客户端
# ============================
try:
    print(f"🌐 Connecting to OpenEnv at {ENV_URL}...")
    client = CalculatorEnv(base_url=ENV_URL)
    # 测试一下连接
    client.reset()
    print("✅ Environment Connected!")
except Exception as e:
    print(f"❌ Environment Connection Failed: {e}")
    print("Please make sure 'python -m calculator_env.server.app' is running.")
    exit()

def run_agent_inference(p1, v1, v2, sure):
    p2 = 100 - p1
    
    # 1. 同步题目参数给环境 (这样如果需要服务端计算Reward才准确，虽然这里只是测试)
    client.set_problem(p1, v1, v2, sure)
    client.reset()
    
    system_prompt = "You are a rational economic agent. Use the <tool> tag to perform python calculations. Finally output 'Final Decision: accept' or 'reject'."
    user_prompt = f"The prospect is: {p1}% chance of ${v1}, {p2}% chance of ${v2}. The sure outcome is: ${sure}. Do you accept?"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"\n🤖 User: {user_prompt}")
    print("-" * 60)
    
    # 模拟多轮交互 (最多 5 轮)
    for turn in range(5):
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # 生成直到遇到 </tool> 或结束
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                stop_strings=["</tool>"], 
                tokenizer=tokenizer,
                do_sample=False # Greedy
            )
            
        # 获取新生成的内容
        new_content = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        print(f"🤖 Agent (Turn {turn}): {new_content}")
        
        text += new_content
        
        # --- 核心：调用 OpenEnv ---
        if "</tool>" in new_content:
            # 构造 Action
            action = CalculatorAction(message=new_content)
            
            # 发送给服务器，获取结果
            try:
                step_result = client.step(action)
                tool_output = step_result.observation.feedback
                
                print(f"🌍 OpenEnv Response: {tool_output.strip()}")
                
                # 将结果拼回去
                text += tool_output
                
            except Exception as e:
                print(f"❌ OpenEnv Error: {e}")
                break
                
        elif "Final Decision" in new_content:
            print("✅ Decision Reached.")
            break
        else:
            print("⚠️ Generation stopped without tool or decision.")
            break

if __name__ == "__main__":
    # 运行测试
    # 50% 2000 vs 800 (应拒绝)
    run_agent_inference(50, 2000, 0, 800)