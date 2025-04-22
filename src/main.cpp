#include <assert.h>
#include <print>
#include <ranges>

#include <onnxruntime_cxx_api.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;


static void AddTensorRT(Ort::SessionOptions& session_options) {
    OrtTensorRTProviderOptions options {};
    options.device_id = 0;
    options.trt_max_workspace_size = 2147483648;
    options.trt_max_partition_iterations = 1000;
    options.trt_min_subgraph_size = 1;
    options.trt_fp16_enable = 1;
    options.trt_int8_enable = 1;
    options.trt_int8_use_native_calibration_table = 1;
    options.trt_dump_subgraphs = 1;
    // below options are strongly recommended !
    options.trt_engine_cache_enable = 1;
    options.trt_engine_cache_path = "I:/tensorrt_cache";
    session_options.AppendExecutionProvider_TensorRT(options);
}


static vector<uint8_t> chw_to_hwc(const float* input, size_t h, size_t w) {
    const size_t stride = h * w;
    vector<uint8_t> output_data(stride * 3);
    for (size_t c = 0; c != 3; ++c) {
        size_t t = c * stride;
        for (size_t i = 0; i != stride; ++i) {
            float f = input[t + i];
            if (f < 0.f || f > 255.0f) f = 0;
            output_data[i * 3 + c] = (uint8_t)f;
        }
    }
    return output_data;
}


struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    void* data = nullptr;

    explicit Image(const char* path) {
        data = stbi_loadf(path, &width, &height, &channels, STBI_rgb);
    }

    ~Image() {
        stbi_image_free(data);
    }
};


struct Model {
    Model(const Ort::Env& env, const wchar_t* model_path) {
        session_options.RegisterCustomOpsLibrary(L"ortextensions.dll", Ort::CustomOpConfigs());
        session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions());

        session = Ort::Session(env, model_path, session_options);
        bindings = Ort::IoBinding(session);

        input_names.resize(session.GetInputCount());
        for (auto [i, name] : input_names | views::enumerate) {
            input_types.emplace_back(session.GetInputTypeInfo(i));
            input_name_ptrs.emplace_back(session.GetInputNameAllocated(i, allocator));
            name = input_name_ptrs.back().get();
        }

        output_names.resize(session.GetOutputCount());
        for (auto [i, name] : output_names | views::enumerate) {
            output_types.emplace_back(session.GetOutputTypeInfo(i));
            output_name_ptrs.emplace_back(session.GetOutputNameAllocated(i, allocator));
            name = output_name_ptrs.back().get();
            //Ort::MemoryInfo output_mem_info(name, OrtDeviceAllocator, 0, OrtMemTypeDefault);
            auto output_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            bindings.BindOutput(name, output_mem_info);
        }
    }

    vector<Ort::Value> Run(span<const char*> input_names, const vector<const Ort::Value*>& input_tensors) {
        for (const auto& [name, value] : views::zip(input_names, input_tensors)) {
            bindings.BindInput(name, *value);
        }
        session.Run(Ort::RunOptions {}, bindings);
        return bindings.GetOutputValues();
    }


    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::IoBinding bindings { nullptr };
    Ort::Session session { nullptr };

    vector<Ort::AllocatedStringPtr> input_name_ptrs;
    vector<const char*> input_names;
    vector<Ort::TypeInfo> input_types;

    vector<Ort::AllocatedStringPtr> output_name_ptrs;
    vector<const char*> output_names;
    vector<Ort::TypeInfo> output_types;

    vector<Ort::Value> output_values;
};

static int run_inference(Model& tokenizer, Model& text_encoder, Model& unet) {
    vector<const char*> prompts { "A bee on a flower.", "A cat approaching the bee." };
    vector<int64_t> input_shape { ssize(prompts) };

    auto input_tensor = Ort::Value::CreateTensor(tokenizer.allocator, input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    input_tensor.FillStringTensor(prompts.data(), prompts.size());

    const auto tokenizer_output = tokenizer.Run(span(&tokenizer.input_names[0], 1), vector<const Ort::Value*>{&input_tensor});

    Ort::TensorTypeAndShapeInfo source_info = tokenizer_output[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = source_info.GetShape();
    size_t total_elements = source_info.GetElementCount();
    const int64_t* source_data = tokenizer_output[0].GetTensorData<int64_t>();
    Ort::Value target_tensor = Ort::Value::CreateTensor(text_encoder.allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    int32_t* target_data = target_tensor.GetTensorMutableData<int32_t>();
    for (size_t i = 0; i < total_elements; ++i) target_data[i] = static_cast<int32_t>(source_data[i]);

    const auto text_encoder_output = text_encoder.Run(text_encoder.input_names, vector<const Ort::Value*>{&target_tensor});

    std::vector<int64_t> ts_shape = { 1 };
    Ort::Value timestep = Ort::Value::CreateTensor(text_encoder.allocator, ts_shape.data(), ts_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    timestep.GetTensorMutableData<int64_t>()[0] = 0;
    unet.Run(unet.input_names, { &tokenizer_output[1], &timestep, &text_encoder_output[0], &text_encoder_output[1] });
    return 0;
}


int main(int argc, char* argv[]) {
    try {
        const auto providers = Ort::GetAvailableProviders();
        println("Available providers:");
        for (const auto& provider : providers) {
            println("{}", provider);
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "");
        Model tokenizer(env, L"I:/huggingface/hub/models--TensorStack--stable-diffusion-3.5-large-turbo-onnx/snapshots/57fe62c728694fcb1c327bc2e884f642631b30ad/tokenizer/model.onnx");
        Model text_encoder(env, L"I:/huggingface/hub/models--TensorStack--stable-diffusion-3.5-large-turbo-onnx/snapshots/57fe62c728694fcb1c327bc2e884f642631b30ad/text_encoder/model.onnx");
        Model unet(env, L"I:/huggingface/hub/models--TensorStack--stable-diffusion-3.5-large-turbo-onnx/snapshots/57fe62c728694fcb1c327bc2e884f642631b30ad/unet/model.onnx");
        Model vae_encoder(env, L"I:/huggingface/hub/models--TensorStack--stable-diffusion-3.5-large-turbo-onnx/snapshots/57fe62c728694fcb1c327bc2e884f642631b30ad/vae_encoder/model.onnx");
        Model vae_decoder(env, L"I:/huggingface/hub/models--TensorStack--stable-diffusion-3.5-large-turbo-onnx/snapshots/57fe62c728694fcb1c327bc2e884f642631b30ad/vae_decoder/model.onnx");
        
        run_inference(tokenizer, text_encoder, unet);
    }
    catch (const Ort::Exception& e) {
        println("Exception ({}): {}", (int)e.GetOrtErrorCode(), e.what());
        return -1;
    }

    return 0;
}
