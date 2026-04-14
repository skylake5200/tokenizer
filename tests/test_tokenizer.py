#!/usr/bin/env python3
"""
对齐 C++ test_tokenizer.cpp 的测试脚本

核心原则：
1. 使用 apply_chat_template（与 C++ 一致）
2. C++ 只在最后一个消息是 USER 时才添加 generation prompt
3. C++ 根据 think_in_prompt 控制是否保留 think 内容
"""

import argparse
import sys
from transformers import AutoTokenizer


def remove_thinking(text: str, end_think_token: str = "</think>") -> str:
    """从文本中移除 think 标签及其内容，对齐 C++ remove_thinking"""
    pos = text.find(end_think_token)
    if pos == -1:
        return text.strip()
    result = text[pos + len(end_think_token):]
    return result.strip()


def print_ids(ids, label="ids"):
    """打印 token ids（无多余逗号）"""
    print(f"{label} size: {len(ids)}")
    print("[", end="")
    for i, id in enumerate(ids):
        if i > 0:
            print(", ", end="")
        print(id, end="")
    print("]")


def print_text(text, label="text"):
    """打印文本"""
    print(f"{label}:\n{text}\n")


def diff_token_ids(ids1, ids2):
    """计算两个 token id 列表的差异"""
    if len(ids1) >= len(ids2):
        return []
    for i in range(len(ids1)):
        if ids1[i] != ids2[i]:
            return []
    return ids2[len(ids1):]


def apply_chat_template_aligned(tokenizer, messages, think_in_prompt=True, add_generation_prompt=None):
    """
    对齐 C++ 的 apply_chat_template 行为：
    1. 只在最后一个消息是 USER 时才添加 generation prompt
    2. 根据 think_in_prompt 控制是否保留 think 内容
    """
    # 对齐 C++: 如果最后一个消息不是 USER，强制 add_generation_prompt=False
    if add_generation_prompt is None:
        add_generation_prompt = (messages[-1]["role"] == "user")
    elif messages[-1]["role"] != "user":
        # C++ 只在最后一个消息是 USER 时才添加
        add_generation_prompt = False
    
    # 对齐 C++ think_in_prompt: 处理 assistant 消息的 content
    processed_messages = []
    for msg in messages:
        if msg["role"] == "assistant" and not think_in_prompt:
            # 移除 think 内容
            cleaned_content = remove_thinking(msg["content"])
            processed_messages.append({"role": "assistant", "content": cleaned_content})
        else:
            processed_messages.append(msg)
    
    return tokenizer.apply_chat_template(
        processed_messages, 
        tokenize=False, 
        add_generation_prompt=add_generation_prompt
    )


def test_text_tokenizer(tokenizer, system_prompt):
    """测试文本 tokenizer - 对齐 C++ test_text_tokenizer"""
    
    # 测试 1: 保留 thinking 内容 (think_in_prompt=True)
    print("\n" + "="*60)
    print("Test 1: think_in_prompt=True")
    print("="*60)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n你好！有什么我可以帮助你的吗？"}
    ]
    
    # 对齐 C++: 最后一个消息是 assistant，不添加 generation prompt
    text = apply_chat_template_aligned(tokenizer, messages, think_in_prompt=True)
    print_text(text)
    
    ids = tokenizer.encode(text)
    print_ids(ids)
    
    decoded = tokenizer.decode(ids)
    print_text(decoded, "decoded")
    
    # 添加新一轮对话 - 对齐 C++: messages.push_back({USER, ...})
    messages.append({"role": "user", "content": "你能做什么"})
    # 最后一个消息是 USER，添加 generation prompt
    text2 = apply_chat_template_aligned(tokenizer, messages, think_in_prompt=True, add_generation_prompt=True)
    print_text(text2, "text2")
    
    ids2 = tokenizer.encode(text2)
    print_ids(ids2, "ids2")
    
    decoded2 = tokenizer.decode(ids2)
    print_text(decoded2, "decoded2")
    
    diff_ids = diff_token_ids(ids, ids2)
    print_ids(diff_ids, "diff_ids")
    
    # 测试 2: 不保留 thinking 内容 (think_in_prompt=False, 使用 remove_thinking)
    print("\n" + "="*60)
    print("Test 2: think_in_prompt=False (with <think> tags)")
    print("="*60)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n你好！有什么我可以帮助你的吗？"}
    ]
    
    # 对齐 C++: think_in_prompt=False 会移除 think 内容
    text = apply_chat_template_aligned(tokenizer, messages, think_in_prompt=False)
    print_text(text)
    
    ids = tokenizer.encode(text)
    print_ids(ids)
    
    decoded = tokenizer.decode(ids)
    print_text(decoded, "decoded")
    
    messages.append({"role": "user", "content": "你能做什么"})
    text2 = apply_chat_template_aligned(tokenizer, messages, think_in_prompt=False, add_generation_prompt=True)
    print_text(text2, "text2")
    
    ids2 = tokenizer.encode(text2)
    print_ids(ids2, "ids2")
    
    decoded2 = tokenizer.decode(ids2)
    print_text(decoded2, "decoded2")
    
    diff_ids = diff_token_ids(ids, ids2)
    print_ids(diff_ids, "diff_ids")
    
    # 测试 3: 不保留 thinking 内容 (assistant 消息以 </think> 开头)
    print("\n" + "="*60)
    print("Test 3: think_in_prompt=False (assistant content starts with </think>)")
    print("="*60)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "</think>\n\n你好！有什么我可以帮助你的吗？"}
    ]
    
    # 对齐 C++: remove_thinking 会处理以 </think> 开头的内容
    text = apply_chat_template_aligned(tokenizer, messages, think_in_prompt=False)
    print_text(text)
    
    ids = tokenizer.encode(text)
    print_ids(ids)
    
    decoded = tokenizer.decode(ids)
    print_text(decoded, "decoded")
    
    messages.append({"role": "user", "content": "你能做什么"})
    text2 = apply_chat_template_aligned(tokenizer, messages, think_in_prompt=False, add_generation_prompt=True)
    print_text(text2, "text2")
    
    ids2 = tokenizer.encode(text2)
    print_ids(ids2, "ids2")
    
    decoded2 = tokenizer.decode(ids2)
    print_text(decoded2, "decoded2")
    
    diff_ids = diff_token_ids(ids, ids2)
    print_ids(diff_ids, "diff_ids")


def test_image_tokenizer(tokenizer, system_prompt):
    """测试多模态 tokenizer - 对齐 C++ test_image_tokenizer"""
    
    print("\n" + "="*60)
    print("Image/Video Tests")
    print("="*60)
    
    # 基础对话（用于 IMAGE/VIDEO 测试）
    messages_base = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
    ]
    
    # IMAGE 测试
    print("\n--- IMAGE input ---")
    
    text = apply_chat_template_aligned(tokenizer, messages_base, think_in_prompt=True)
    print_text(text)
    ids = tokenizer.encode(text)
    print_ids(ids)
    
    decoded = tokenizer.decode(ids)
    print_text(decoded, "decoded")
    
    # 添加图片输入 - 使用模型支持的方式
    chat_template = tokenizer.chat_template or ""
    tokenizer_name = tokenizer.__class__.__name__.lower()
    
    try:
        if "gemma" in tokenizer_name or "<start_of_image>" in chat_template:
            messages_img = messages_base + [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "帮我看看这张图片"}]}]
        elif "qwen" in tokenizer_name or "<|vision_start|>" in chat_template:
            messages_img = messages_base + [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "帮我看看这张图片"}]}]
        else:
            # MiniCPM 格式
            messages_img = messages_base + [{"role": "user", "content": "<image>\n帮我看看这张图片"}]
        
        text_img = apply_chat_template_aligned(tokenizer, messages_img, think_in_prompt=True)
        print_text(text_img, "text (with image)")
        
        ids2 = tokenizer.encode(text_img)
        print_ids(ids2)
        
        decoded2 = tokenizer.decode(ids2)
        print_text(decoded2, "decoded")
        
        print(f"Token count increased: {len(ids)} -> {len(ids2)} (+{len(ids2) - len(ids)})")
    except Exception as e:
        print(f"Image test skipped or failed: {e}")
    
    # VIDEO 测试
    print("\n--- VIDEO input ---")
    try:
        if "gemma" in tokenizer_name or "<start_of_image>" in chat_template:
            video_content = [{"type": "image"} for _ in range(8)] + [{"type": "text", "text": "帮我看看这个视频"}]
            messages_video = messages_base + [{"role": "user", "content": video_content}]
        elif "qwen" in tokenizer_name:
            video_text = "<image>" * 8 + "\n帮我看看这个视频"
            messages_video = messages_base + [{"role": "user", "content": video_text}]
        else:
            messages_video = messages_base + [{"role": "user", "content": "<video>\n帮我看看这个视频"}]
        
        text_video = apply_chat_template_aligned(tokenizer, messages_video, think_in_prompt=True)
        print_text(text_video, "text (with video)")
        
        ids3 = tokenizer.encode(text_video)
        print_ids(ids3)
        
        decoded3 = tokenizer.decode(ids3)
        print_text(decoded3, "decoded")
        
        print(f"Token count: {len(ids3)}")
    except Exception as e:
        print(f"Video test skipped or failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Tokenizer test (aligned with C++)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                       help='HF model path (e.g., Qwen/Qwen3-1.7B)')
    parser.add_argument('--no-media', action='store_true',
                       help='Skip image/video tests')
    args = parser.parse_args()
    
    # 加载 tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
    except:
        tokenizer = None
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        except Exception as e:
            # Workaround for some newer tokenizer configs (e.g. google/gemma-4-31B-it) where
            # `extra_special_tokens` may not match Transformers' expected schema.
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, extra_special_tokens={})
    
    if tokenizer is None:
        print("Failed to load tokenizer")
        sys.exit(1)
    
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Vocab size: {len(tokenizer.get_vocab())}")
    
    # system_prompt 必须与 C++ 完全一致（注意：没有句号！）
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant"
    
    # 运行文本测试
    test_text_tokenizer(tokenizer, system_prompt)
    
    # 运行多模态测试
    if not args.no_media:
        test_image_tokenizer(tokenizer, system_prompt)
    
    print("\n" + "="*60)
    print("Tests completed")
    print("="*60)


if __name__ == '__main__':
    main()
