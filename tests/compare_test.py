#!/usr/bin/env python3
"""
对比 C++ 和 Python 测试输出是否一致
"""

import subprocess
import sys
import json

def run_cpp_test(tokenizer_path, model_type):
    """运行 C++ 测试并解析输出"""
    cmd = ["./build/test_tokenizer", "-t", tokenizer_path, "-m", str(model_type), "--no-media", "-v"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    tests = []
    current_test = {}
    
    for line in result.stdout.split('\n'):
        if '[TEST]' in line:
            if current_test:
                tests.append(current_test)
            current_test = {'name': line.split(']')[1].strip()}
        elif 'ids size:' in line and 'ids2' not in line:
            current_test['ids_size'] = int(line.split(':')[1].strip())
        elif 'ids2' in line and 'size:' in line:
            current_test['ids2_size'] = int(line.split(':')[1].strip())
        elif 'diff_ids size:' in line:
            current_test['diff_size'] = int(line.split(':')[1].strip())
    
    if current_test:
        tests.append(current_test)
    
    return tests

def parse_python_output(output):
    """解析 Python 测试输出"""
    tests = []
    current_test = {}
    
    for line in output.split('\n'):
        if line.startswith('Test '):
            if current_test and 'ids_size' in current_test:
                tests.append(current_test)
            current_test = {'name': line.split(':')[0].strip()}
        elif 'ids size:' in line and 'ids2' not in line.lower():
            current_test['ids_size'] = int(line.split(':')[1].strip())
        elif 'ids2 size:' in line:
            current_test['ids2_size'] = int(line.split(':')[1].strip())
        elif 'diff_ids size:' in line:
            current_test['diff_size'] = int(line.split(':')[1].strip())
    
    if current_test and 'ids_size' in current_test:
        tests.append(current_test)
    
    return tests

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare C++ and Python test outputs')
    parser.add_argument('--tokenizer-path', type=str, required=True)
    parser.add_argument('--model-type', type=int, required=True)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Comparing C++ and Python test outputs")
    print("=" * 60)
    
    # 运行 C++ 测试
    print("\nRunning C++ test...")
    cpp_tests = run_cpp_test(args.tokenizer_path, args.model_type)
    
    # 运行 Python 测试
    print("Running Python test...")
    cmd = ["python", "tests/test_tokenizer.py", "--tokenizer_path", args.tokenizer_path, "--no-media"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    python_tests = parse_python_output(result.stdout)
    
    # 对比结果
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    all_match = True
    for i, (cpp, py) in enumerate(zip(cpp_tests, python_tests)):
        print(f"\nTest {i+1}: {cpp['name']}")
        print(f"  C++:  ids={cpp.get('ids_size', 'N/A')}, ids2={cpp.get('ids2_size', 'N/A')}, diff={cpp.get('diff_size', 'N/A')}")
        print(f"  Python: ids={py.get('ids_size', 'N/A')}, ids2={py.get('ids2_size', 'N/A')}, diff={py.get('diff_size', 'N/A')}")
        
        match = (cpp.get('ids_size') == py.get('ids_size') and 
                 cpp.get('ids2_size') == py.get('ids2_size') and
                 cpp.get('diff_size') == py.get('diff_size'))
        
        if match:
            print("  ✓ MATCH")
        else:
            print("  ✗ MISMATCH")
            all_match = False
    
    print("\n" + "=" * 60)
    if all_match:
        print("✓ All tests match!")
        return 0
    else:
        print("✗ Some tests mismatch!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
