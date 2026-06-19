#include "kvision_inference.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <cassert>

namespace kvision {

// ─────────────────────────────────────────────
// Pathology labels
// ─────────────────────────────────────────────

static const char* kPathologyLabels[] = {
    "Normal",
    "Edema",
    "Tumor (Glioma)",
    "Tumor (Meningioma)",
    "Hemorrhage",
    "Ischemia",
    "MS Lesions",
    "Atrophy",
    "Hydrocephalus",
    "AVM",
    "Abscess"
};

const char* pathology_label(Pathology p) {
    int idx = static_cast<int>(p);
    if (idx < 0 || idx > 10) return "Unknown";
    return kPathologyLabels[idx];
}

// ─────────────────────────────────────────────
// PIMPL Implementation
// ─────────────────────────────────────────────

struct InferenceEngine::Impl {
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    InferenceConfig config;

    // Cached input/output metadata
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

    Impl(const InferenceConfig& cfg)
        : env(ORT_LOGGING_LEVEL_WARNING, "KVisionInference"),
          session(nullptr),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          config(cfg)
    {
        // ── Session options ──
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(cfg.num_threads);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.enable_profiling) {
            session_opts.EnableProfiling("kvision_profile");
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
                std::cout << "[KVision] CUDA execution provider enabled (device "
                          << cfg.cuda_device_id << ")" << std::endl;
            } catch (const Ort::Exception& e) {
                std::cerr << "[KVision] WARNING: Failed to enable CUDA provider: "
                          << e.what() << std::endl;
                std::cerr << "[KVision] Falling back to CPU." << std::endl;
            }
        }

        // ── Load model ──
        session = Ort::Session(env, cfg.model_path.c_str(), session_opts);

        // ── Extract input/output metadata ──
        Ort::AllocatorWithDefaultOptions allocator;

        auto in_name = session.GetInputNameAllocated(0, allocator);
        input_name = in_name.get();

        auto out_name = session.GetOutputNameAllocated(0, allocator);
        output_name = out_name.get();

        auto in_type_info = session.GetInputTypeInfo(0);
        auto in_tensor_info = in_type_info.GetTensorTypeAndShapeInfo();
        input_shape = in_tensor_info.GetShape();

        auto out_type_info = session.GetOutputTypeInfo(0);
        auto out_tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
        output_shape = out_tensor_info.GetShape();

        // Log model info
        std::cout << "[KVision] Model loaded: " << cfg.model_path << std::endl;
        std::cout << "[KVision] Input: '" << input_name << "' shape=[";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i + 1 < input_shape.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "[KVision] Output: '" << output_name << "' shape=[";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i + 1 < output_shape.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    InferenceResult run(const float* data, size_t num_elements) {
        // Validate input size
        if (static_cast<int64_t>(num_elements) != KSpaceDims::total_elements) {
            std::ostringstream oss;
            oss << "Input size mismatch: expected " << KSpaceDims::total_elements
                << " elements, got " << num_elements;
            throw std::invalid_argument(oss.str());
        }

        // Build input tensor (shape: [1, 64, 1, 128, 128, 2])
        std::vector<int64_t> shape = {
            KSpaceDims::batch, KSpaceDims::slices, KSpaceDims::channels,
            KSpaceDims::height, KSpaceDims::width, KSpaceDims::ri
        };

        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(data),
            num_elements,
            shape.data(),
            shape.size()
        );

        // Prepare name arrays
        const char* input_names[] = { input_name.c_str() };
        const char* output_names[] = { output_name.c_str() };

        // ── Run inference ──
        auto t_start = std::chrono::high_resolution_clock::now();

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // ── Parse output ──
        float* logits_ptr = output_tensors[0].GetTensorMutableData<float>();

        InferenceResult result{};
        result.inference_time_ms = elapsed_ms;

        // Copy logits
        for (int i = 0; i < 11; ++i) {
            result.logits[i] = logits_ptr[i];
        }

        // Softmax
        float max_logit = *std::max_element(result.logits.begin(), result.logits.end());
        float sum_exp = 0.0f;
        for (int i = 0; i < 11; ++i) {
            result.probabilities[i] = std::exp(result.logits[i] - max_logit);
            sum_exp += result.probabilities[i];
        }
        for (int i = 0; i < 11; ++i) {
            result.probabilities[i] /= sum_exp;
        }

        // Argmax
        int best = 0;
        for (int i = 1; i < 11; ++i) {
            if (result.probabilities[i] > result.probabilities[best]) {
                best = i;
            }
        }
        result.predicted_class = static_cast<Pathology>(best);
        result.confidence = result.probabilities[best];

        return result;
    }
};

// ─────────────────────────────────────────────
// InferenceEngine public API
// ─────────────────────────────────────────────

InferenceEngine::InferenceEngine(const InferenceConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

InferenceEngine::~InferenceEngine() = default;
InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;

InferenceResult InferenceEngine::infer(const std::vector<float>& kspace_data) {
    return pimpl_->run(kspace_data.data(), kspace_data.size());
}

InferenceResult InferenceEngine::infer(const float* kspace_data, size_t num_elements) {
    return pimpl_->run(kspace_data, num_elements);
}

std::string InferenceEngine::get_model_description() const {
    std::ostringstream oss;
    oss << "KVision Fused S4-CNN MRI Classifier\n"
        << "  Model: " << pimpl_->config.model_path << "\n"
        << "  Input: " << pimpl_->input_name << " [1, 64, 1, 128, 128, 2] float32\n"
        << "  Output: " << pimpl_->output_name << " [1, 11] float32\n"
        << "  Backend: " << (pimpl_->config.use_cuda ? "CUDA" : "CPU") << "\n"
        << "  Classes: 11 pathology types";
    return oss.str();
}

} // namespace kvision
