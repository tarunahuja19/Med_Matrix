# ⚙️ C++ Inference Engine (`ai-service/inference`)

This folder contains the **high-performance native C++ inference engines** for MedMatrix, built using **ONNX Runtime (ORT) v1.24.4**. These engines wrap exported ONNX networks to enable GPU-accelerated (CUDA) and multi-threaded CPU inference.

---

## 🏎️ Performance Benchmarks

Inference times measured on production-equivalent workloads:

### 1. Volumetric S4-CNN Pathology Classifier (`fused_model_128.onnx`)
*   **Input Dimensions:** `[1, 8, 16, 128, 128, 2]` (Batch, Slices, Coils, H, W, Real/Imaginary).
*   **Total flat float elements:** 4,194,304.

| Backend | Avg Inference Time | Throughput |
| :--- | :---: | :---: |
| **CPU** (4 Threads) | 79.2 ms | ~12.6 inf/sec |
| **CUDA (GPU)** | 11.8 ms | ~84.8 inf/sec |

### 2. State Space (SSM) K-Space Anomaly Estimator (`anomaly_detector.onnx`)
*   **Input Dimensions:** `kspace`: `[1, 32, 256, 256]` ( stacked real/imaginary parts of 16 coils) and `contrast`: `[1]` (T1/T2 weighted).
*   **Throughput:** Processes slices at **~23.4 ms per inference** (~42.8 slices/sec) on a single CPU thread.

---

## 🗺️ File Map

| File Name | Purpose |
| :--- | :--- |
| [CMakeLists.txt](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/CMakeLists.txt) | Build configuration importing `onnxruntime`, `abseil`, and `protobuf`. |
| [kvision_inference.h](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/kvision_inference.h) | Header declaring `InferenceEngine`, `InferenceResult`, pathology enums, and shape constants. |
| [kvision_inference.cpp](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/kvision_inference.cpp) | Implementation: ORT session configuration, CUDA/CPU EP registration, softmax and argmax post-processing. |
| [anomaly_detector_inference.h](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/anomaly_detector_inference.h) | Header declaring `AnomalyDetectorEngine` and result struct. |
| [anomaly_detector_inference.cpp](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/anomaly_detector_inference.cpp) | Implementation: manages dual-input bindings (K-space float data + contrast int64 index). |
| [test_inference.cpp](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/test_inference.cpp) | 10-test suite verifying loading, bounds, determinism, sensitivity, and benchmarking. |
| [test_anomaly_detector_inference.cpp](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/test_anomaly_detector_inference.cpp) | 9-test suite verifying SSM regression bounds, accuracy, and performance. |

---

## 🛠️ Build & Compilation Instructions

### Prerequisites (Arch Linux / EndeavourOS)
Ensure the required libraries are installed:
```bash
sudo pacman -S onnxruntime-cuda abseil-cpp protobuf cmake
```
*(For CPU-only environments, install `onnxruntime` instead).*

### Compilation Steps
1.  Navigate to the inference directory:
    ```bash
    cd ai-service/inference
    ```
2.  Create and compile the build artifacts:
    ```bash
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc)
    ```

### Running Test Suites
Verify that both compiled test suites compile and execute successfully:
```bash
./test_inference
./test_anomaly_detector_inference
```

---

## 📝 Integration Code Examples

### 1. S4-CNN Volumetric Classification
```cpp
#include "kvision_inference.h"
#include <vector>

int main() {
    kvision::Config cfg;
    cfg.model_path = "../fused_model_128.onnx";
    cfg.num_threads = 4;
    cfg.use_gpu = true; // Attempt CUDA EP initialization

    // Initialize Engine
    kvision::InferenceEngine engine(cfg);

    // Prepare dummy volumetric complex input (4,194,304 floats)
    std::vector<float> input_kspace(kvision::KSpaceDims::total_elements, 0.0f);

    // Run inference
    kvision::InferenceResult result = engine.infer(input_kspace);

    std::cout << "Predicted Class: " << static_cast<int>(result.predicted_class) << "\n";
    std::cout << "Confidence: " << result.confidence * 100.0f << "%\n";
    std::cout << "Execution Time: " << result.inference_time_ms << " ms\n";
    return 0;
}
```

### 2. SSM K-Space Anomaly Detection
```cpp
#include "anomaly_detector_inference.h"
#include <vector>

int main() {
    kvision::AnomalyConfig cfg;
    cfg.model_path = "../anomaly_detector.onnx";
    cfg.num_threads = 1;

    // Initialize Engine
    kvision::AnomalyDetectorEngine engine(cfg);

    // Prepare dummy slice input (32 channels * 256 * 256 = 2,097,152 floats)
    std::vector<float> slice_data(kvision::AnomalyKSpaceDims::total_elements, 0.0f);
    int64_t contrast = 0; // T1 Weighted

    // Run inference
    kvision::AnomalyResult result = engine.infer(slice_data, contrast);

    std::cout << "Noise Severity: " << result.noise_severity << "\n";
    std::cout << "Motion Severity: " << result.motion_severity << "\n";
    std::cout << "Phase Severity: " << result.phase_severity << "\n";
    return 0;
}
```
