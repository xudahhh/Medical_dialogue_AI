import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_from_disk, DatasetDict
# --------------------- 测试配置 ---------------------
base_model = "models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # 原始基座模型路径
lora_weights = "./result/lora_v2results/checkpoint-7000"       # LoRA适配器路径

# 保持与训练一致的量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # 二次量化进一步压缩
)


# --------------------- 模型加载 ---------------------
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token  # 确保pad token设置正确

# 加载基础模型（需与训练时配置一致）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, lora_weights)
model = model.merge_and_unload()  # 合并权重（可选，方便后续部署）

# --------------------- 批量测试（使用测试集） ---------------------
def evaluate_testset():
    # 加载测试集（假设已处理为相同格式）
    test_dataset = load_from_disk("./medical_instructions")
    
    total = len(test_dataset)
    correct = 0
    i = 0
    for example in test_dataset:
        if i >10:
            break
        i+=1
        prompt = f"医学问题：{example['input']}\n回答："  # 保持训练时的prompt格式
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7,        # 控制生成随机性
                repetition_penalty=1.1  # 防止重复生成
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = response.split("回答：")[-1].strip()
        
        # 简单匹配评估（可根据需求替换为更复杂的评估逻辑）
        if generated_answer.strip() == example['output'].strip():
            correct += 1
    
    print(f"测试集准确率: {correct/total:.2%}")

# --------------------- 单条样本测试 ---------------------
def interactive_test():
    while True:
        question = input("\n请输入医学问题（输入q退出）:")
        if question.lower() == 'q':
            break
            
        prompt = f"医学问题：{question}\n回答："
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.9,
            top_k = 50,
            repetition_penalty = 1.2, # 重复惩罚系数
            temperature=0.7
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n模型回答：", response.split("回答：")[-1].strip())

# --------------------- 执行测试 ---------------------
if __name__ == "__main__":
    model.eval()  # 设置为评估模式
    
    # 执行批量测试
    # evaluate_testset()
    # 交互式测试
    interactive_test()