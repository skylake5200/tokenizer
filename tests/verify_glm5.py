#!/usr/bin/env python3
"""
验证 GLM-5 Tokenizer 的 Python 和 C++ 实现是否一致
"""

import sys
import os
import subprocess
import tempfile

# 加载 Python tokenizer
from tokenizers import Tokenizer

def load_tokenizer(tokenizer_path):
    """加载 GLM-5 tokenizer"""
    return Tokenizer.from_file("/tmp/glm5_tokenizer.json")

def apply_chat_template_python(messages, add_generation_prompt=True):
    """Python 版本的 chat template 实现"""
    result = "[gMASK]<sop>"
    
    # Find last user index
    last_user_index = -1
    for i, m in enumerate(messages):
        if m['role'] == 'user':
            last_user_index = i
    
    for i, m in enumerate(messages):
        if m['role'] == 'system':
            result += f"<|system|>{m['content']}"
        elif m['role'] == 'user':
            result += f"<|user|>{m['content']}"
        elif m['role'] == 'assistant':
            result += "<|assistant|>"
            
            # Handle thinking content
            content = m['content']
            reasoning_content = ''
            
            # Extract reasoning from <think> tags
            if '<think>' in content and '</think>' in content:
                parts = content.split('</think>')
                if len(parts) >= 2:
                    think_part = parts[0]
                    if '<think>' in think_part:
                        reasoning_content = think_part.split('<think>')[-1].strip()
                    content = '</think>'.join(parts[1:]).lstrip('\n')
            
            # For simplicity, always use </think> for history
            result += "</think>"
            
            if content.strip():
                result += content
    
    if add_generation_prompt:
        result += "<|assistant|><think>"
    
    return result

def test_with_cpp(tokenizer_path, model_type, messages, add_generation_prompt=True):
    """使用 C++ 测试程序验证"""
    # 构建输入
    input_lines = []
    for m in messages:
        if m['role'] == 'system':
            input_lines.append(f"/system {m['content']}")
        elif m['role'] == 'user':
            input_lines.append(f"/user {m['content']}")
        elif m['role'] == 'assistant':
            input_lines.append(f"/assistant {m['content']}")
    input_lines.append("/encode")
    input_lines.append("")
    
    input_text = "\n".join(input_lines)
    
    # 运行 C++ 程序
    cmd = ["./build/test_tokenizer", "-t", tokenizer_path, "-m", str(model_type)]
    result = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    
    return result.stdout, result.stderr

def main():
    tokenizer_path = "tests/assets/glm5_tokenizer.txt"
    model_type = 13  # GLM5
    
    print("=" * 60)
    print("GLM-5 Tokenizer 验证")
    print("=" * 60)
    
    # 加载 tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    
    # 测试用例
    test_cases = [
        {
            "name": "基本对话",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！很高兴见到你。"}
            ]
        },
        {
            "name": "多轮对话",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ]
        },
        {
            "name": "带 thinking 的对话",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "<think>Let me calculate...</think>The answer is 4."}
            ]
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        print(f"\n测试: {test['name']}")
        print("-" * 40)
        
        messages = test['messages']
        
        # Python 实现
        python_text = apply_chat_template_python(messages, add_generation_prompt=True)
        python_ids = tokenizer.encode(python_text).ids
        
        print(f"Python text: {repr(python_text[:100])}...")
        print(f"Python IDs (first 20): {python_ids[:20]}")
        
        # 运行 C++ 测试
        stdout, stderr = test_with_cpp(tokenizer_path, model_type, messages)
        
        # 解析 C++ 输出
        cpp_ids = None
        for line in stderr.split('\n'):
            if 'encode ids:' in line:
                try:
                    ids_str = line.split('encode ids:')[1].strip()
                    cpp_ids = [int(x) for x in ids_str.split() if x.isdigit()]
                except:
                    pass
        
        if cpp_ids:
            print(f"C++ IDs (first 20): {cpp_ids[:20]}")
            
            # 比较
            if python_ids == cpp_ids:
                print("✓ Python 和 C++ 的 token IDs 匹配!")
            else:
                print("✗ 不匹配!")
                print(f"  Python length: {len(python_ids)}")
                print(f"  C++ length: {len(cpp_ids)}")
                
                # 找到第一个不同的位置
                for i, (p, c) in enumerate(zip(python_ids, cpp_ids)):
                    if p != c:
                        print(f"  第一个不同位置: {i}, Python: {p}, C++: {c}")
                        break
                all_passed = False
        else:
            print("无法从 C++ 输出中解析 IDs")
            print(f"C++ stderr: {stderr[:500]}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败!")
    print("=" * 60)

if __name__ == "__main__":
    main()
