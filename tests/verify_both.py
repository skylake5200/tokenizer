#!/usr/bin/env python3
"""
验证 GLM-5 和 Kimi-K2.5 Tokenizer 的 Python 和 C++ 实现是否一致
"""

import sys
import os
import subprocess
import tempfile
import base64

# 添加模块路径
module_path = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/moonshotai/Kimi-K2.5/3367c8d1c68584429fab7faf845a32d5195b6ac1")
if module_path not in sys.path:
    sys.path.insert(0, module_path)

def load_glm5_tokenizer():
    """加载 GLM-5 tokenizer"""
    from tokenizers import Tokenizer
    return Tokenizer.from_file("/tmp/glm5_tokenizer.json")

def load_kimik25_tokenizer():
    """加载 Kimi-K2.5 tokenizer"""
    import tool_declaration_ts
    sys.modules['transformers_modules.moonshotai.Kimi-K2.tool_declaration_ts'] = tool_declaration_ts
    
    tokenization_code = open(os.path.join(module_path, 'tokenization_kimi.py')).read()
    tokenization_code = tokenization_code.replace('from .tool_declaration_ts import', 'from tool_declaration_ts import')
    exec(tokenization_code, globals())
    
    return TikTokenTokenizer.from_pretrained("moonshotai/Kimi-K2.5", trust_remote_code=True)

def apply_chat_template_glm5(messages, add_generation_prompt=True):
    """GLM-5 chat template"""
    result = "[gMASK]<sop>"
    
    for m in messages:
        if m['role'] == 'system':
            result += f"<|system|>{m['content']}"
        elif m['role'] == 'user':
            result += f"<|user|>{m['content']}"
        elif m['role'] == 'assistant':
            result += f"<|assistant|>{m['content']}"
    
    if add_generation_prompt:
        result += "<|assistant|><think>"
    
    return result

def apply_chat_template_kimik25(messages, add_generation_prompt=True):
    """Kimi-K2.5 chat template - 简化版"""
    result = ""
    
    for m in messages:
        if m['role'] == 'system':
            result += f"<|im_system|>system<|im_middle|>{m['content']}<|im_end|>"
        elif m['role'] == 'user':
            result += f"<|im_user|>user<|im_middle|>{m['content']}<|im_end|>"
        elif m['role'] == 'assistant':
            result += f"<|im_assistant|>assistant<|im_middle|<think></think>{m['content']}<|im_end|>"
    
    if add_generation_prompt:
        result += "<|im_assistant|>assistant<|im_middle|<think>"
    
    return result

def test_model(name, model_type, tokenizer_path, apply_template_func, load_tokenizer_func):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    # 加载 tokenizer
    tokenizer = load_tokenizer_func()
    
    # 测试用例
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    # Python 实现
    python_text = apply_template_func(messages, add_generation_prompt=True)
    print(f"\nPython template:\n{repr(python_text)}")
    
    if hasattr(tokenizer, 'encode'):
        if name == "GLM-5":
            python_encoding = tokenizer.encode(python_text)
            python_ids = python_encoding.ids
        else:
            python_ids = tokenizer.encode(python_text)
    else:
        print("Tokenizer has no encode method")
        return False
    
    print(f"Python IDs: {python_ids}")
    
    # 测试特殊 token 编码
    if name == "GLM-5":
        test_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
    else:
        test_tokens = ["<|im_system|>", "<|im_user|>", "<|im_assistant|>", "<|im_end|>"]
    
    print(f"\n特殊 token 编码:")
    for token in test_tokens:
        if name == "GLM-5":
            encoded = tokenizer.encode(token).ids
        else:
            encoded = tokenizer.encode(token)
        print(f"  {repr(token)}: {encoded}")
    
    return True

def main():
    # 测试 GLM-5
    test_model(
        "GLM-5",
        13,
        "tests/assets/glm5_tokenizer.txt",
        apply_chat_template_glm5,
        load_glm5_tokenizer
    )
    
    # 测试 Kimi-K2.5
    test_model(
        "Kimi-K2.5",
        15,
        "tests/assets/kimi_k25_tokenizer.txt",
        apply_chat_template_kimik25,
        load_kimik25_tokenizer
    )

if __name__ == "__main__":
    main()
