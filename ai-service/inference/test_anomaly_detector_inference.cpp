/**
 * test_anomaly_detector_inference.cpp
 *
 * Tests the AnomalyDetector C++ ONNX inference engine.
 */

#include "anomaly_detector_inference.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <random>
#include <numeric>
#include <cassert>
#include <string>
#include <sstream>

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

std::vector<float> generate_synthetic_anomaly_kspace(unsigned seed = 42) {
    const size_t total = kvision::AnomalyKSpaceDims::total_elements;
    std::vector<float> data(total);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

bool test_model_loading(const std::string& model_path) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 1: Model Loading ═══" CLR_RESET << std::endl;

    kvision::AnomalyConfig cfg;
    cfg.model_path = model_path;
    cfg.use_cuda = false;
    cfg.num_threads = 1;

    try {
        kvision::AnomalyDetectorEngine engine(cfg);
        std::string desc = engine.get_model_description();
        TEST_ASSERT(!desc.empty(), "Model description is non-empty");
        std::cout << "\n  Model Info:\n";
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

bool test_basic_inference(kvision::AnomalyDetectorEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 2: Basic Inference ═══" CLR_RESET << std::endl;

    auto kspace = generate_synthetic_anomaly_kspace(42);
    auto result = engine.infer(kspace, 0); // T1 contrast

    TEST_ASSERT(result.noise_severity >= 0.0f && result.noise_severity <= 1.0f,
                "Noise severity is bounded in [0, 1] (got " + std::to_string(result.noise_severity) + ")");
    TEST_ASSERT(result.motion_severity >= 0.0f && result.motion_severity <= 1.0f,
                "Motion severity is bounded in [0, 1] (got " + std::to_string(result.motion_severity) + ")");
    TEST_ASSERT(result.phase_severity >= 0.0f && result.phase_severity <= 1.0f,
                "Phase severity is bounded in [0, 1] (got " + std::to_string(result.phase_severity) + ")");

    std::cout << "\n  Inference Results:\n";
    std::cout << "    Noise Severity:  " << std::fixed << std::setprecision(4) << result.noise_severity << "\n";
    std::cout << "    Motion Severity: " << result.motion_severity << "\n";
    std::cout << "    Phase Severity:  " << result.phase_severity << "\n";
    std::cout << "    Inference time:  " << std::setprecision(2) << result.inference_time_ms << " ms\n";

    return true;
}

bool test_determinism(kvision::AnomalyDetectorEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 3: Determinism ═══" CLR_RESET << std::endl;

    auto kspace = generate_synthetic_anomaly_kspace(123);
    auto result1 = engine.infer(kspace, 1); // T2 contrast
    auto result2 = engine.infer(kspace, 1);

    TEST_ASSERT(std::fabs(result1.noise_severity - result2.noise_severity) < 1e-5f, "Noise severity is deterministic");
    TEST_ASSERT(std::fabs(result1.motion_severity - result2.motion_severity) < 1e-5f, "Motion severity is deterministic");
    TEST_ASSERT(std::fabs(result1.phase_severity - result2.phase_severity) < 1e-5f, "Phase severity is deterministic");

    return true;
}

bool test_invalid_input(kvision::AnomalyDetectorEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 4: Invalid Input Handling ═══" CLR_RESET << std::endl;

    std::vector<float> bad_data(100, 0.0f);
    bool threw = false;
    try {
        engine.infer(bad_data, 0);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "    (caught: " << e.what() << ")\n";
    }
    TEST_ASSERT(threw, "Throws std::invalid_argument for wrong input size");

    return true;
}

bool test_throughput(kvision::AnomalyDetectorEngine& engine) {
    std::cout << "\n" CLR_BOLD CLR_CYAN "═══ Test 5: Throughput Benchmark ═══" CLR_RESET << std::endl;

    auto kspace = generate_synthetic_anomaly_kspace(42);
    const int num_runs = 5;
    double total_ms = 0.0;

    // Warmup
    engine.infer(kspace, 0);

    for (int i = 0; i < num_runs; ++i) {
        auto result = engine.infer(kspace, 0);
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

int main(int argc, char** argv) {
    std::cout << CLR_BOLD "\n╔══════════════════════════════════════════════════╗\n"
              << "║  Anomaly Detector C++ Inference Engine Suite    ║\n"
              << "╚══════════════════════════════════════════════════╝\n" CLR_RESET;

    std::string model_path = "../anomaly_detector.onnx";
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

    if (!test_model_loading(model_path)) {
        std::cerr << CLR_RED "\nCannot proceed — model failed to load.\n" CLR_RESET;
        return 1;
    }

    kvision::AnomalyConfig cfg;
    cfg.model_path = model_path;
    cfg.use_cuda = use_cuda;
    cfg.num_threads = 1;

    kvision::AnomalyDetectorEngine engine(cfg);

    test_basic_inference(engine);
    test_determinism(engine);
    test_invalid_input(engine);
    test_throughput(engine);

    std::cout << "\n" CLR_BOLD "═══════════════════════════════════════════\n"
              << "  Results: " CLR_GREEN << tests_passed << " passed" CLR_RESET
              << ", " CLR_RED << tests_failed << " failed" CLR_RESET
              << " (" << (tests_passed + tests_failed) << " total)\n"
              << CLR_BOLD "═══════════════════════════════════════════\n" CLR_RESET;

    return tests_failed > 0 ? 1 : 0;
}
