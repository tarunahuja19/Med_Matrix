/**
 * test_inference.cpp
 *
 * Tests the KVision ONNX inference engine with synthetic K-space data.
 * Verifies:
 *   1. Model loads successfully
 *   2. Inference produces valid output shape (11 classes)
 *   3. Probabilities sum to ~1.0
 *   4. Predicted class is in valid range [0, 10]
 *   5. Timing is reasonable
 *   6. Multiple inferences produce consistent results
 */

#include "kvision_inference.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <random>
#include <numeric>
#include <cassert>
#include <string>

// ─────────────────────────────────────────────
// ANSI color codes for pretty output
// ─────────────────────────────────────────────
#define CLR_RESET   "\033[0m"
#define CLR_GREEN   "\033[32m"
#define CLR_RED     "\033[31m"
#define CLR_YELLOW  "\033[33m"
#define CLR_CYAN    "\033[36m"
#define CLR_BOLD    "\033[1m"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << CLR_RED "  ✗ FAIL: " CLR_RESET << msg << std::endl; \
        tests_failed++; \
        return false; \
    } else { \
        std::cout << CLR_GREEN "  ✓ PASS: " CLR_RESET << msg << std::endl; \
        tests_passed++; \
    } \
} while(0)

// ─────────────────────────────────────────────
// Generate synthetic K-space data
// ─────────────────────────────────────────────
std::vector<float> generate_synthetic_kspace(unsigned seed = 42) {
    const size_t total = kvision::KSpaceDims::total_elements;
    std::vector<float> data(total);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

// ─────────────────────────────────────────────
// Test: Model Loading
// ─────────────────────────────────────────────
bool test_model_loading(const std::string& model_path) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 1: Model Loading ═══" CLR_RESET << std::endl;

    kvision::InferenceConfig cfg;
    cfg.model_path = model_path;
    cfg.use_cuda = false;
    cfg.num_threads = 4;

    try {
        kvision::InferenceEngine engine(cfg);
        std::string desc = engine.get_model_description();
        TEST_ASSERT(!desc.empty(), "Model description is non-empty");
        std::cout << "\n  Model Info:\n";
        // Indent each line
        std::istringstream iss(desc);
        std::string line;
        while (std::getline(iss, line)) {
            std::cout << "    " << line << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << CLR_RED "  ✗ FAIL: Model loading threw: " CLR_RESET << e.what() << std::endl;
        tests_failed++;
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────
// Test: Basic Inference
// ─────────────────────────────────────────────
bool test_basic_inference(kvision::InferenceEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 2: Basic Inference ═══" CLR_RESET << std::endl;

    auto kspace = generate_synthetic_kspace(42);
    auto result = engine.infer(kspace);

    // Check predicted class is valid
    int cls = static_cast<int>(result.predicted_class);
    TEST_ASSERT(cls >= 0 && cls <= 10, "Predicted class in valid range [0, 10]");

    // Check confidence is valid probability
    TEST_ASSERT(result.confidence > 0.0f && result.confidence <= 1.0f,
                "Confidence is a valid probability");

    // Check all 11 logits are finite
    bool all_finite = true;
    for (int i = 0; i < 11; ++i) {
        if (!std::isfinite(result.logits[i])) { all_finite = false; break; }
    }
    TEST_ASSERT(all_finite, "All 11 logits are finite");

    // Check probabilities sum to ~1.0
    float prob_sum = 0.0f;
    for (int i = 0; i < 11; ++i) prob_sum += result.probabilities[i];
    TEST_ASSERT(std::fabs(prob_sum - 1.0f) < 1e-4f,
                "Softmax probabilities sum to ~1.0 (got " + std::to_string(prob_sum) + ")");

    // Print results
    std::cout << "\n  Inference Results:\n";
    std::cout << "    Predicted: " CLR_BOLD << kvision::pathology_label(result.predicted_class)
              << CLR_RESET " (class " << cls << ")\n";
    std::cout << "    Confidence: " << std::fixed << std::setprecision(2)
              << result.confidence * 100.0f << "%\n";
    std::cout << "    Inference time: " << std::setprecision(1) << result.inference_time_ms << " ms\n";
    std::cout << "\n    Class probabilities:\n";
    for (int i = 0; i < 11; ++i) {
        const char* label = kvision::pathology_label(static_cast<kvision::Pathology>(i));
        float pct = result.probabilities[i] * 100.0f;
        std::cout << "      [" << std::setw(2) << i << "] " << std::setw(22) << std::left
                  << label << std::right << " : " << std::setprecision(2) << std::setw(6)
                  << pct << "% ";
        // Mini bar chart
        int bar_len = static_cast<int>(pct / 2.0f);
        for (int b = 0; b < bar_len; ++b) std::cout << "█";
        std::cout << "\n";
    }

    return true;
}

// ─────────────────────────────────────────────
// Test: Deterministic (same input → same output)
// ─────────────────────────────────────────────
bool test_determinism(kvision::InferenceEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 3: Determinism ═══" CLR_RESET << std::endl;

    auto kspace = generate_synthetic_kspace(123);
    auto result1 = engine.infer(kspace);
    auto result2 = engine.infer(kspace);

    TEST_ASSERT(result1.predicted_class == result2.predicted_class,
                "Same input → same predicted class");

    float logit_diff = 0.0f;
    for (int i = 0; i < 11; ++i) {
        logit_diff += std::fabs(result1.logits[i] - result2.logits[i]);
    }
    TEST_ASSERT(logit_diff < 1e-3f,
                "Same input → same logits (diff=" + std::to_string(logit_diff) + ")");

    return true;
}

// ─────────────────────────────────────────────
// Test: Different inputs produce different outputs
// ─────────────────────────────────────────────
bool test_sensitivity(kvision::InferenceEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 4: Input Sensitivity ═══" CLR_RESET << std::endl;

    auto kspace_a = generate_synthetic_kspace(1);
    auto kspace_b = generate_synthetic_kspace(999);

    auto result_a = engine.infer(kspace_a);
    auto result_b = engine.infer(kspace_b);

    float logit_diff = 0.0f;
    for (int i = 0; i < 11; ++i) {
        logit_diff += std::fabs(result_a.logits[i] - result_b.logits[i]);
    }
    TEST_ASSERT(logit_diff > 0.01f,
                "Different inputs → different logits (diff=" + std::to_string(logit_diff) + ")");

    return true;
}

// ─────────────────────────────────────────────
// Test: Invalid input size
// ─────────────────────────────────────────────
bool test_invalid_input(kvision::InferenceEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 5: Invalid Input Handling ═══" CLR_RESET << std::endl;

    std::vector<float> bad_data(100, 0.0f);
    bool threw = false;
    try {
        engine.infer(bad_data);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "    (caught: " << e.what() << ")\n";
    }
    TEST_ASSERT(threw, "Throws std::invalid_argument for wrong input size");

    return true;
}

// ─────────────────────────────────────────────
// Test: Throughput benchmark
// ─────────────────────────────────────────────
bool test_throughput(kvision::InferenceEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 6: Throughput Benchmark ═══" CLR_RESET << std::endl;

    auto kspace = generate_synthetic_kspace(42);
    const int num_runs = 5;
    double total_ms = 0.0;

    // Warmup
    engine.infer(kspace);

    for (int i = 0; i < num_runs; ++i) {
        auto result = engine.infer(kspace);
        total_ms += result.inference_time_ms;
    }

    double avg_ms = total_ms / num_runs;
    TEST_ASSERT(avg_ms > 0.0, "Average inference time is positive (" +
                std::to_string(avg_ms) + " ms)");

    std::cout << "\n    Throughput: " << std::fixed << std::setprecision(1)
              << avg_ms << " ms/inference avg over " << num_runs << " runs\n";
    std::cout << "    ~" << std::setprecision(1) << (1000.0 / avg_ms)
              << " inferences/sec\n";

    return true;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    std::cout << CLR_BOLD "\n╔══════════════════════════════════════════════════╗\n"
              << "║   KVision C++ Inference Engine — Test Suite       ║\n"
              << "╚══════════════════════════════════════════════════╝\n" CLR_RESET;

    // Parse args
    std::string model_path = "../fused_model.onnx";
    bool use_cuda = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--model path.onnx] [--cuda]\n";
            return 0;
        }
    }

    std::cout << "\n  Model: " << model_path << "\n";
    std::cout << "  Backend: " << (use_cuda ? "CUDA" : "CPU") << "\n";

    // Test 1: Loading
    if (!test_model_loading(model_path)) {
        std::cerr << CLR_RED "\nCannot proceed — model failed to load.\n" CLR_RESET;
        return 1;
    }

    // Create engine for remaining tests
    kvision::InferenceConfig cfg;
    cfg.model_path = model_path;
    cfg.use_cuda = use_cuda;
    cfg.num_threads = 4;

    kvision::InferenceEngine engine(cfg);

    // Run all tests
    test_basic_inference(engine);
    test_determinism(engine);
    test_sensitivity(engine);
    test_invalid_input(engine);
    test_throughput(engine);

    // Summary
    std::cout << "\n" CLR_BOLD "═══════════════════════════════════════════\n"
              << "  Results: " CLR_GREEN << tests_passed << " passed" CLR_RESET
              << ", " CLR_RED << tests_failed << " failed" CLR_RESET
              << " (" << (tests_passed + tests_failed) << " total)\n"
              << CLR_BOLD "═══════════════════════════════════════════\n" CLR_RESET;

    return tests_failed > 0 ? 1 : 0;
}
