#pragma once

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <stdexcept>

// Forward declare to avoid leaking ONNX Runtime headers
namespace Ort { class Session; class Env; class SessionOptions; class MemoryInfo; }

namespace kvision {

/// Pathology class labels (11 classes matching training)
enum class Pathology : int {
    Normal            = 0,
    Edema             = 1,
    Tumor_Glioma      = 2,
    Tumor_Meningioma  = 3,
    Hemorrhage        = 4,
    Ischemia          = 5,
    MS_Lesions        = 6,
    Atrophy           = 7,
    Hydrocephalus     = 8,
    AVM               = 9,
    Abscess           = 10
};

/// Human-readable label for a pathology
const char* pathology_label(Pathology p);

/// Result of a single inference
struct InferenceResult {
    Pathology predicted_class;
    float     confidence;              ///< softmax probability of top class
    std::array<float, 11> logits;      ///< raw logit scores
    std::array<float, 11> probabilities; ///< softmax probabilities
    double    inference_time_ms;       ///< wall-clock inference time
};

/// Configuration for the inference engine
struct InferenceConfig {
    std::string model_path;                ///< path to .onnx file
    bool        use_cuda       = false;    ///< enable CUDA execution provider
    int         cuda_device_id = 0;        ///< GPU device index
    int         num_threads    = 4;        ///< intra-op parallelism (CPU)
    bool        enable_profiling = false;  ///< ONNX Runtime profiling
};

/// K-space volume dimensions expected by the model
struct KSpaceDims {
    static constexpr int64_t batch    = 1;
    static constexpr int64_t slices   = 64;
    static constexpr int64_t channels = 1;
    static constexpr int64_t height   = 128;
    static constexpr int64_t width    = 128;
    static constexpr int64_t ri       = 2;   // real + imaginary
    static constexpr int64_t total_elements = batch * slices * channels * height * width * ri;
};

/**
 * @brief ONNX Runtime inference engine for the Fused S4-CNN MRI classifier.
 *
 * Usage:
 *   kvision::InferenceConfig cfg;
 *   cfg.model_path = "fused_model.onnx";
 *   cfg.use_cuda = true;
 *
 *   kvision::InferenceEngine engine(cfg);
 *
 *   // Prepare K-space data: [64, 1, 128, 128, 2] float32
 *   std::vector<float> kspace_data = load_kspace(...);
 *   auto result = engine.infer(kspace_data);
 *
 *   std::cout << "Predicted: " << kvision::pathology_label(result.predicted_class)
 *             << " (" << result.confidence * 100 << "%)" << std::endl;
 */
class InferenceEngine {
public:
    /// Construct and load the ONNX model
    explicit InferenceEngine(const InferenceConfig& config);
    ~InferenceEngine();

    // Non-copyable, movable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;

    /// Run inference on a K-space volume
    /// @param kspace_data  Flat float array of size KSpaceDims::total_elements
    ///                     Layout: [S=64][C=1][H=128][W=128][RI=2]
    ///                     where [RI=0] = real part, [RI=1] = imaginary part
    /// @return InferenceResult with predicted class and probabilities
    InferenceResult infer(const std::vector<float>& kspace_data);

    /// Run inference on raw pointer (avoids copy)
    InferenceResult infer(const float* kspace_data, size_t num_elements);

    /// Get model metadata
    std::string get_model_description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace kvision
