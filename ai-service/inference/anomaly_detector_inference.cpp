#include "anomaly_detector_inference.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <cassert>

namespace kvision {

struct AnomalyDetectorEngine::Impl {
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    AnomalyConfig config;

    // Cached input/output metadata
    std::string kspace_input_name;
    std::string contrast_input_name;
    std::string output_name;

    Impl(const AnomalyConfig& cfg)
        : env(ORT_LOGGING_LEVEL_WARNING, "KVisionAnomalyInference"),
          session(nullptr),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          config(cfg)
    {
        // ── Session options ──
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(cfg.num_threads);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.enable_profiling) {
            session_opts.EnableProfiling("anomaly_profile");
        }

        // ── CUDA provider ──
        if (cfg.use_cuda) {
            try {
                OrtCUDAProviderOptions cuda_opts;
                cuda_opts.device_id = cfg.cuda_device_id;
                cuda_opts.arena_extend_strategy = 0;  // kNextPowerOfTwo
                cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_opts.do_copy_in_default_stream = 1;
                session_opts.AppendExecutionProvider_CUDA(cuda_opts);
                std::cout << "[AnomalyDetector] CUDA execution provider enabled (device "
                          << cfg.cuda_device_id << ")" << std::endl;
            } catch (const Ort::Exception& e) {
                std::cerr << "[AnomalyDetector] WARNING: Failed to enable CUDA provider: "
                          << e.what() << std::endl;
                std::cerr << "[AnomalyDetector] Falling back to CPU." << std::endl;
            }
        }

        // ── Load model ──
        session = Ort::Session(env, cfg.model_path.c_str(), session_opts);

        // ── Extract input/output metadata ──
        Ort::AllocatorWithDefaultOptions allocator;

        // Model has two inputs: "kspace" (0) and "contrast" (1)
        auto in_name0 = session.GetInputNameAllocated(0, allocator);
        kspace_input_name = in_name0.get();

        auto in_name1 = session.GetInputNameAllocated(1, allocator);
        contrast_input_name = in_name1.get();

        auto out_name = session.GetOutputNameAllocated(0, allocator);
        output_name = out_name.get();

        std::cout << "[AnomalyDetector] Model loaded: " << cfg.model_path << std::endl;
        std::cout << "[AnomalyDetector] Input 0: '" << kspace_input_name << "'" << std::endl;
        std::cout << "[AnomalyDetector] Input 1: '" << contrast_input_name << "'" << std::endl;
        std::cout << "[AnomalyDetector] Output: '" << output_name << "'" << std::endl;
    }

    AnomalyResult run(const float* kspace_data, size_t num_elements, int64_t contrast) {
        // Validate input size
        if (static_cast<int64_t>(num_elements) != AnomalyKSpaceDims::total_elements) {
            std::ostringstream oss;
            oss << "Input kspace size mismatch: expected " << AnomalyKSpaceDims::total_elements
                << " elements, got " << num_elements;
            throw std::invalid_argument(oss.str());
        }

        // 1. Build kspace tensor (shape: [1, 32, 256, 256])
        std::vector<int64_t> kspace_shape = {
            AnomalyKSpaceDims::batch, AnomalyKSpaceDims::channels,
            AnomalyKSpaceDims::height, AnomalyKSpaceDims::width
        };
        auto kspace_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(kspace_data),
            num_elements,
            kspace_shape.data(),
            kspace_shape.size()
        );

        // 2. Build contrast tensor (shape: [1])
        std::vector<int64_t> contrast_shape = {
            AnomalyKSpaceDims::batch
        };
        // Store contrast in a local variable to preserve lifetime during Run
        int64_t local_contrast = contrast;
        auto contrast_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            &local_contrast,
            1,
            contrast_shape.data(),
            contrast_shape.size()
        );

        // Prepare inputs and names
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(kspace_tensor));
        input_tensors.push_back(std::move(contrast_tensor));

        const char* input_names[] = { kspace_input_name.c_str(), contrast_input_name.c_str() };
        const char* output_names[] = { output_name.c_str() };

        // ── Run inference ──
        auto t_start = std::chrono::high_resolution_clock::now();

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            2,
            output_names,
            1
        );

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // ── Parse output (shape: [1, 3]) ──
        float* pred_ptr = output_tensors[0].GetTensorMutableData<float>();

        AnomalyResult result{};
        result.noise_severity  = pred_ptr[0];
        result.motion_severity = pred_ptr[1];
        result.phase_severity  = pred_ptr[2];
        result.inference_time_ms = elapsed_ms;

        return result;
    }
};

AnomalyDetectorEngine::AnomalyDetectorEngine(const AnomalyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

AnomalyDetectorEngine::~AnomalyDetectorEngine() = default;
AnomalyDetectorEngine::AnomalyDetectorEngine(AnomalyDetectorEngine&&) noexcept = default;
AnomalyDetectorEngine& AnomalyDetectorEngine::operator=(AnomalyDetectorEngine&&) noexcept = default;

AnomalyResult AnomalyDetectorEngine::infer(const std::vector<float>& kspace_data, int64_t contrast) {
    return pimpl_->run(kspace_data.data(), kspace_data.size(), contrast);
}

AnomalyResult AnomalyDetectorEngine::infer(const float* kspace_data, size_t num_elements, int64_t contrast) {
    return pimpl_->run(kspace_data, num_elements, contrast);
}

std::string AnomalyDetectorEngine::get_model_description() const {
    std::ostringstream oss;
    oss << "AnomalyDetector SSM MRI Estimator\n"
        << "  Model:   " << pimpl_->config.model_path << "\n"
        << "  Input 0: " << pimpl_->kspace_input_name << " [1, 32, 256, 256] float32\n"
        << "  Input 1: " << pimpl_->contrast_input_name << " [1] int64\n"
        << "  Output:  " << pimpl_->output_name << " [1, 3] float32 (noise, motion, phase)\n"
        << "  Backend: " << (pimpl_->config.use_cuda ? "CUDA" : "CPU");
    return oss.str();
}

} // namespace kvision
