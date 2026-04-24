#include "../include/BaseTokenizer.hpp"
#include <iostream>
#include <vector>
#include <string>

// ANSI colors
namespace color {
    const char* reset = "\033[0m";
    const char* red = "\033[31m";
    const char* green = "\033[32m";
    const char* yellow = "\033[33m";
    const char* bold = "\033[1m";
}

struct TestCase {
    std::string name;
    bool should_work;
};

int main() {
    printf("%s%s=== Test create_tokenizer(std::string) ===%s\n\n", 
           color::bold, color::yellow, color::reset);
    
    // 测试用例：有效的模型名称
    std::vector<TestCase> valid_names = {
        {"Qwen2_5", true},
        {"Qwen3", true},
        {"Qwen3VL", true},
        {"InternVL3", true},
        {"HunYuan", true},
        {"SmolVLM2", true},
        {"FastVLM", true},
        {"MiniCPM4", true},
        {"MiniCPMV4", true},
        {"Gemma3", true},
        {"Gemma3VL", true},
        {"SmolLM2", true},
        {"SmolLM3", true},
        {"GLM5", true},
        {"GLM5VL", true},
        {"KimiK25", true},
        {"KimiK25VL", true},
        {"Qwen3Omni", true},
        {"Qwen3_5", true},
        {"Qwen3_5VL", true},
        {"MiniMaxM2", true},
        {"MiniMaxM2VL", true},
        {"MiniCPMO4_5", true},
        {"InternVL3_5", true},
        {"Gemma4", true},
        {"Gemma4VL", true},
    };
    
    // 测试用例：无效的模型名称
    std::vector<TestCase> invalid_names = {
        {"qwen3", false},       // 大小写不匹配
        {"Qwen4", false},       // 不存在
        {"Unknown", false},     // 不存在
        {"", false},            // 空字符串
    };
    
    int passed = 0;
    int failed = 0;
    
    // 测试有效名称
    printf("%sTesting VALID names:%s\n", color::bold, color::reset);
    for (const auto& test : valid_names) {
        auto tokenizer = create_tokenizer(test.name);
        bool success = (tokenizer != nullptr);
        
        if (success == test.should_work) {
            printf("  %s✓%s %-15s -> %s%s%s\n", 
                   color::green, color::reset,
                   test.name.c_str(),
                   color::green, "SUCCESS", color::reset);
            passed++;
        } else {
            printf("  %s✗%s %-15s -> %s%s%s (expected: %s)\n", 
                   color::red, color::reset,
                   test.name.c_str(),
                   color::red, "FAILED", color::reset,
                   test.should_work ? "SUCCESS" : "FAILED");
            failed++;
        }
    }
    
    // 测试无效名称
    printf("\n%sTesting INVALID names:%s\n", color::bold, color::reset);
    for (const auto& test : invalid_names) {
        auto tokenizer = create_tokenizer(test.name);
        bool success = (tokenizer != nullptr);
        
        if (success == test.should_work) {
            printf("  %s✓%s %-15s -> %s%s%s\n", 
                   color::green, color::reset,
                   test.name.empty() ? "(empty)" : test.name.c_str(),
                   color::green, success ? "SUCCESS" : "NULL (expected)", color::reset);
            passed++;
        } else {
            printf("  %s✗%s %-15s -> %s%s%s (unexpected)\n", 
                   color::red, color::reset,
                   test.name.empty() ? "(empty)" : test.name.c_str(),
                   color::red, success ? "SUCCESS" : "NULL", color::reset);
            failed++;
        }
    }
    
    // 对比测试：字符串 vs 枚举
    printf("\n%sComparing string vs enum create:%s\n", color::bold, color::reset);
    
    auto tokenizer_by_string = create_tokenizer(std::string("Qwen3"));
    auto tokenizer_by_enum = create_tokenizer(Qwen3);
    
    if (tokenizer_by_string && tokenizer_by_enum) {
        printf("  Both created successfully ✓\n");
        
        // 测试它们行为是否一致（如果能加载 tokenizer 文件）
        // 这里只是验证创建成功
    } else {
        printf("  %sMismatch:%s string=%p, enum=%p\n", 
               color::red, color::reset,
               (void*)tokenizer_by_string.get(), 
               (void*)tokenizer_by_enum.get());
    }
    
    // 总结
    printf("\n%s=== Summary ===%s\n", color::bold, color::reset);
    printf("  Passed: %s%d%s\n", color::green, passed, color::reset);
    printf("  Failed: %s%d%s\n", color::red, failed, color::reset);
    printf("  Total:  %d\n", passed + failed);
    
    return failed > 0 ? 1 : 0;
}
