# 环境安装 (确保使用最新库)
# pip install -U transformers datasets peft accelerate bitsandbytes trl
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig  # 新增量化配置
)
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_from_disk, DatasetDict
import numpy as np

# --------------------- 显存优化配置 ---------------------
# 8位量化配置 (显存降低约2倍)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # 二次量化进一步压缩
)

# --------------------- 模型加载 ---------------------
model_name = "models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# --------------------- 模型加载 ---------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 关键步骤：准备4位量化模型以进行训练
model = prepare_model_for_kbit_training(model)

# --------------------- LoRA配置优化 ---------------------
lora_config = LoraConfig(
    r=8,              # 降低秩维度（原16过高）
    lora_alpha=32,    # 保持alpha/r=4的比例
    target_modules=[
        "q_proj", "v_proj",          # 仅保留关键注意力层
        "up_proj", "down_proj"       # 增加FFN层适配
    ],  
    lora_dropout=0.2,               # 增大dropout防止过拟合
    modules_to_save=["embed_tokens", "lm_head"],  # 关键层全参数训练
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------------- 数据处理优化 ---------------------

def format_prompt(example):
    """优化后的格式化函数（不再需要seen_inputs参数）"""
    return {"text": f"医学问题：{example['input']}\n回答：{example['output']}"}

def remove_duplicates(dataset):
    """使用哈希值进行高效去重"""
    seen = set()
    
    def _filter_duplicates(example):
        h = hash(example["input"].strip())  # 对输入内容取哈希值
        if h not in seen:
            seen.add(h)
            return True
        return False
    
    return dataset.filter(_filter_duplicates)

# 加载数据集（优化版本）
def load_data(path):
    # 原始数据加载
    dataset = load_from_disk(path)
    
    # 全局去重（先于数据划分）
    deduplicated_dataset = remove_duplicates(dataset)
    # 数据划分
    split_dataset = deduplicated_dataset.train_test_split(
        test_size=0.1
    )
    
    train_data = split_dataset["train"].select(range(len(split_dataset["train"])//5))
    test_data = split_dataset["test"].select(range(len(split_dataset["test"])//5))
    
    return DatasetDict({
        "train": train_data.map(format_prompt),
        "test": test_data.map(format_prompt)
    })

# 使用示例
dataset = load_data("./medical_instructions")

# --------------------- 训练参数优化 ---------------------
training_args = TrainingArguments(
    output_dir="./result/lora_v3results",
    logging_dir="./result/logs_v3",
    per_device_train_batch_size=2,      # 增大批次大小（提升4倍）
    gradient_accumulation_steps=4,      # 降低梯度累积步数（总batch=16）
    learning_rate=5e-5,                 # 适当提高学习率
    fp16=False,                         # 关闭FP16
    bf16=True,                          # 启用BF16（4090支持更好）
    gradient_checkpointing=True,       # 关闭检查点（显存充足时提速）
    optim="adamw_torch_fused",          # 使用融合优化器
    save_steps=10000,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    report_to=["tensorboard"],
    warmup_steps=100,                   # 添加训练预热
    num_train_epochs=300,                # 大幅减少训练轮次（原100轮严重过拟合）
    warmup_ratio=0.1,                  # 改用比例式预热（更适应不同数据量）
    weight_decay=0.01,                 # 添加权重衰减
    max_grad_norm=0.5,                 # 收紧梯度裁剪（原1.0过大）
    lr_scheduler_type="linear",         # 简单线性衰减更稳定
)


# --------------------- 训练器配置 ---------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],       # 添加验证集监控
    tokenizer=tokenizer,
    args=training_args
)

# --------------------- 开始训练 ---------------------
trainer.train()

# 保存适配器权重
model.save_pretrained("lora_adapter")