from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# 将路径插入到 sys.path 的开头
sys.path.insert(0, '/root/transformers-debug/src')
base_model_path = "/root/autodl-fs/Qwen2-1.5B-Instruct"
draft_model_path = "/root/autodl-fs/Qwen2-1.5B-Instruct"


def load_and_generate(model_name: str, prompt: str, max_length: int = 50):
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path)
    # 将输入文本编码为模型的输入格式
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    # 生成文本
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,  # 设置为1以生成单个序列
        do_sample=True,          # 启用采样以增加生成的多样性
        temperature=0.7,         # 控制生成的随机性
        top_k=50,                # 使用 top-k 采样
        top_p=0.95,
        assistant_model=draft_model
    )

    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    model_name = "gpt2"  # 选择一个预训练模型
    prompt = "Once upon a time"
    generated_text = load_and_generate(model_name, prompt)
    print("Generated text:", generated_text)
