#pragma once

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

// Forward declare to avoid leaking ONNX Runtime headers
namespace Ort { class Session; class Env; class SessionOptions; class MemoryInfo; }

namespace kvision {

/// Result of anomaly estimation inference
struct AnomalyResult {
    float  noise_severity;   ///< continuous regression estimate [0, 1]
    float  motion_severity;  ///< continuous regression estimate [0, 1]
    float  phase_severity;   ///< continuous regression estimate [0, 1]
    double inference_time_ms; ///< wall-clock execution time
};

/// Configuration for the anomaly detector engine
struct AnomalyConfig {
    std::string model_path;                ///< path to anomaly_detector.onnx file
    bool        use_cuda       = false;    ///< enable CUDA execution provider
    int         cuda_device_id = 0;        ///< GPU device index
    int         num_threads    = 1;        ///< intra-op parallelism (CPU)
    bool        enable_profiling = false;  ///< ONNX Runtime profiling
};

/// Anomaly detector K-space dimensions expected per slice
struct AnomalyKSpaceDims {
    static constexpr int64_t batch      = 1;
    static constexpr int64_t channels   = 32;   // 2 * 16 coils (real + imag parts stacked)
    static constexpr int64_t height     = 256;
    static constexpr int64_t width      = 256;
    static constexpr int64_t total_elements = batch * channels * height * width;
};

/**
 * @brief ONNX Runtime inference engine for the State Space Model (SSM) based MRI Anomaly Estimator.
 */
class AnomalyDetectorEngine {
public:
    /// Construct and load the ONNX anomaly detection model
    explicit AnomalyDetectorEngine(const AnomalyConfig& config);
    ~AnomalyDetectorEngine();

    // Non-copyable, movable
    AnomalyDetectorEngine(const AnomalyDetectorEngine&) = delete;
    AnomalyDetectorEngine& operator=(const AnomalyDetectorEngine&) = delete;
    AnomalyDetectorEngine(AnomalyDetectorEngine&&) noexcept;
    AnomalyDetectorEngine& operator=(AnomalyDetectorEngine&&) noexcept;

    /// Run inference on a K-space slice
    /// @param kspace_data  Flat float array of size AnomalyKSpaceDims::total_elements
    ///                     Layout: [channels=32][H=256][W=256] where the 32 channels
    ///                     are real parts of 16 coils followed by imaginary parts.
    /// @param contrast     Contrast type (0 for T1, 1 for T2)
    /// @return AnomalyResult with continuous estimated parameter severities
    AnomalyResult infer(const std::vector<float>& kspace_data, int64_t contrast);

    /// Run inference on raw pointer (avoids copy)
    AnomalyResult infer(const float* kspace_data, size_t num_elements, int64_t contrast);

    /// Get model metadata description
    std::string get_model_description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace kvision
