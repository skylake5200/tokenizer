#!/usr/bin/env python3
"""
通用 Tokenizer 验证脚本
分析 Python (transformers) 的 chat template 行为，输出用于对比 C++ 实现

用法:
    python verify_tokenizer.py --model <hf_model_path> [--test-image] [--test-video]

示例:
    # 验证纯文本模型
    python verify_tokenizer.py --model openbmb/MiniCPM4-0.5B
    
    # 验证多模态模型
    python verify_tokenizer.py --model openbmb/MiniCPM-V-4 --test-image
    
    # 保存结果到文件
    python verify_tokenizer.py --model google/gemma-3-1b-it > gemma3_reference.txt
"""

import argparse
import sys
import os
from typing import List, Tuple
from transformers import AutoTokenizer


def print_section(title: str, char: str = "="):
    """打印带分隔符的标题"""
    print(f"\n{char*60}")
    print(title)
    print(f"{char*60}")


def test_basic_text(tokenizer, system_prompt: str = "You are a helpful assistant.") -> Tuple[List[int], str]:
    """测试基本文本对话"""
    print_section("测试 1: 基本文本对话 (带 generation prompt)")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "你好"},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text)
    
    print(f"\n生成的文本 ({len(text)} 字符):")
    print(repr(text))
    print(f"\nToken IDs ({len(ids)} 个):")
    print(ids)
    
    # 逐 token 解码查看
    print(f"\n逐 token 解码 (前 20 个):")
    for i, tid in enumerate(ids[:20]):
        decoded = tokenizer.decode([tid])
        print(f"  {i:3d}: {tid:6d} -> {repr(decoded)}")
    
    return ids, text


def test_with_assistant_response(tokenizer) -> Tuple[List[int], str]:
    """测试包含 assistant 回复的对话"""
    print_section("测试 2: 完整对话 (含 assistant 回复)")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！很高兴见到你。"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text)
    
    print(f"\n生成的文本 ({len(text)} 字符):")
    print(repr(text))
    print(f"\nToken IDs ({len(ids)} 个):")
    print(ids)
    
    return ids, text


def test_multi_turn(tokenizer) -> Tuple[List[int], str]:
    """测试多轮对话"""
    print_section("测试 3: 多轮对话")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！很高兴见到你。"},
        {"role": "user", "content": "你能做什么？"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text)
    
    print(f"\n生成的文本 ({len(text)} 字符):")
    print(repr(text[:500]))
    if len(text) > 500:
        print("...")
    print(f"\nToken IDs ({len(ids)} 个):")
    print(ids[:50])
    if len(ids) > 50:
        print("...")
    
    return ids, text


def test_image(tokenizer, image_placeholder: str = "<image>") -> Tuple[List[int], str]:
    """测试图片输入"""
    print_section("测试 4: 图片输入")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{image_placeholder}\n帮我看看这张图片"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text)
    
    print(f"\n生成的文本 ({len(text)} 字符):")
    print(repr(text))
    print(f"\nToken IDs ({len(ids)} 个):")
    print(ids)
    
    # 检查图片 token
    vocab = tokenizer.get_vocab()
    image_token_id = vocab.get(image_placeholder.strip('<>'), None) or vocab.get(image_placeholder, None)
    
    print(f"\n图片 token 分析:")
    print(f"  占位符: {image_placeholder}")
    if image_token_id:
        print(f"  Token ID: {image_token_id}")
        if image_token_id in ids:
            print(f"  ✓ 存在于序列中")
        else:
            print(f"  ✗ 未在序列中找到")
    else:
        # 尝试查找包含 image 的 token
        image_tokens = [(k, v) for k, v in vocab.items() if 'image' in k.lower() or 'img' in k.lower()]
        if image_tokens:
            print(f"  可能的图片 tokens:")
            for k, v in image_tokens[:5]:
                print(f"    {k}: {v}")
    
    return ids, text


def analyze_chat_template(tokenizer):
    """分析 chat template 结构"""
    print_section("Chat Template 分析")
    
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        template = tokenizer.chat_template
        print(f"\n原始模板 ({len(template)} 字符):")
        print(template)
        
        # 提取关键模式
        patterns = []
        if '<|im_start|>' in template:
            patterns.append("IM 风格: <|im_start|>...<|im_end|>")
        if '<start_of_turn>' in template:
            patterns.append("Gemma 风格: <start_of_turn>...<end_of_turn>")
        if '<|begin_of_text|>' in template:
            patterns.append("Llama 风格: <|begin_of_text|>...")
        if '{% for message in messages %}' in template:
            patterns.append("使用 Jinja2 for 循环")
        if 'add_generation_prompt' in template:
            patterns.append("支持 add_generation_prompt")
        
        print(f"\n检测到的模式:")
        for p in patterns:
            print(f"  - {p}")
    else:
        print("\n无 chat template，使用默认格式")


def analyze_special_tokens(tokenizer):
    """分析特殊 token"""
    print_section("特殊 Token 分析")
    
    vocab = tokenizer.get_vocab()
    
    print(f"\n基本特殊 Tokens:")
    print(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    print(f"\n特殊 Tokens Map:")
    for k, v in tokenizer.special_tokens_map.items():
        print(f"  {k}: {v}")
    
    # 查找其他可能的特殊 tokens
    special_patterns = ['<|', '|>', '<start', '<end', '<image', '<video', '[INST]', '[/INST]']
    print(f"\n其他可能的特殊 Tokens (包含 {special_patterns}):")
    for token, tid in list(vocab.items())[:10000]:  # 只检查前 10000 个避免太慢
        for pat in special_patterns:
            if pat in token:
                print(f"  {token}: {tid}")
                break


def main():
    parser = argparse.ArgumentParser(
        description='验证 Tokenizer 实现 - 输出 Python 参考结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础验证
  python verify_tokenizer.py --model openbmb/MiniCPM4-0.5B
  
  # 多模态验证
  python verify_tokenizer.py --model openbmb/MiniCPM-V-4 --test-image
  
  # 保存结果用于对比
  python verify_tokenizer.py --model google/gemma-3-1b-it > reference.txt
        """
    )
    parser.add_argument('--model', type=str, required=True, 
                       help='HuggingFace 模型路径 (例如: openbmb/MiniCPM4-0.5B)')
    parser.add_argument('--test-image', action='store_true',
                       help='测试图片输入 (用于多模态模型)')
    parser.add_argument('--test-video', action='store_true',
                       help='测试视频输入 (用于多模态模型)')
    parser.add_argument('--image-placeholder', type=str, default='<image>',
                       help='图片占位符 (默认: <image>)')
    parser.add_argument('--system-prompt', type=str, default='You are a helpful assistant.',
                       help='System prompt 内容')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Tokenizer 验证: {args.model}")
    print("=" * 60)
    
    # 加载 tokenizer
    print(f"\n正在加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        # Workaround for some newer tokenizer configs (e.g. google/gemma-4-31B-it) where
        # `extra_special_tokens` may not match Transformers' expected schema.
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, extra_special_tokens={})
            print(f"警告: 默认加载失败，已使用 extra_special_tokens={{}} 进行兼容加载: {e}")
        except Exception as e2:
            print(f"错误: 加载 tokenizer 失败: {e2}")
            sys.exit(1)
    
    # 基本信息
    vocab = tokenizer.get_vocab()
    print(f"\n基本信息:")
    print(f"  Tokenizer 类型: {type(tokenizer).__name__}")
    print(f"  词表大小: {len(vocab)}")
    
    # 分析
    analyze_chat_template(tokenizer)
    analyze_special_tokens(tokenizer)
    
    # 运行测试
    print_section("测试用例输出", "=")
    
    # 测试 1: 基本文本
    test_basic_text(tokenizer, args.system_prompt)
    
    # 测试 2: 完整对话
    test_with_assistant_response(tokenizer)
    
    # 测试 3: 多轮对话
    test_multi_turn(tokenizer)
    
    # 测试 4: 图片输入
    if args.test_image:
        test_image(tokenizer, args.image_placeholder)
    
    # 总结
    print_section("验证指南", "=")
    print("""
下一步:
1. 确认上述 Python 输出符合预期
2. 运行 C++ 测试: cd build && ./test_tokenizer -t <tokenizer.txt> -m <model_type>
3. 对比 C++ 输出的文本和 Token IDs 是否与 Python 一致
4. 如果不一致，检查 C++ 的 apply_chat_template 实现

常见问题:
- 文本不匹配: 检查特殊字符（换行、空格）和 token 顺序
- IDs 不匹配: 检查 BOS/EOS 是否添加正确
- 长度不同: 检查是否缺少 generation prompt
    """)
    
    print("\n✓ 验证数据生成完成")


if __name__ == "__main__":
    main()
