#include <assert.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <print>
#include <random>
#include <ranges>

#include <onnxruntime_cxx_api.h>
#include <ort_genai.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "pndmscheduler.h"


using namespace std;

static constexpr int64_t image_w = 512;
static constexpr int64_t image_h = 512;


template<typename T>
static consteval ONNXTensorElementDataType GetONNXType() {
    if constexpr (is_same_v<T, float>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if constexpr (is_same_v<T, Ort::Float16_t>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}


template<integral... Idx>
static constexpr pair<size_t, size_t> calc_slice(span<const int64_t> shape, Idx... idx) {
    const auto size = reduce(shape.begin() + sizeof...(Idx), shape.end(), 1LL, multiplies());
    size_t stride = size;
    size_t i = sizeof...(idx) - 1;
    size_t offset = 0;
    (stride = ... = (offset += idx * stride, stride *= shape[i--]));
    return { offset, size };
}


template<typename T, integral... Idx>
static span<const T> TensorToSpan(const Ort::Value& tensor, Idx... idx) {
    const auto [offset, size] = calc_slice(tensor.GetTensorTypeAndShapeInfo().GetShape(), idx...);
    return span(tensor.GetTensorData<T>() + offset, size);
}


template<typename T, typename... Idx>
static span<T> TensorToSpan(Ort::Value& tensor, Idx... idx) {
    const auto [offset, size] = calc_slice(tensor.GetTensorTypeAndShapeInfo().GetShape(), idx...);
    return span(tensor.GetTensorMutableData<T>() + offset, size);
}


template<typename T>
static Ort::Value CopyTensor(const Ort::Value& tensor, const Ort::AllocatorWithDefaultOptions& allocator) {
    const auto info = tensor.GetTensorTypeAndShapeInfo();
    auto new_tensor = Ort::Value::CreateTensor(allocator,
                                               info.GetShape().data(),
                                               info.GetShape().size(),
                                               info.GetElementType());
    memcpy(new_tensor.GetTensorMutableData<T>(), tensor.GetTensorData<T>(), sizeof(T) * info.GetElementCount());
    return new_tensor;
}


template<typename T>
static void RandomizeData(span<T> data, optional<uint32_t> seed = {}) {
    std::random_device rd;
    std::mt19937 gen(seed ? *seed : rd());
    std::normal_distribution<float> dist(0.f, 1.f);

    for (auto& val : data) val = static_cast<T>(dist(gen));
}


template<typename T>
static void ConvertToImgAndSave(span<const T> img, int w, int h, int comps, const string& filename) {
    const auto ch_size = w * h;

    vector<uint8_t> out_img(img.size());

    for (const auto [ch, channel_pixels] : img | views::chunk(ch_size) | views::enumerate) {
        for (const auto [idx, in] : channel_pixels | views::enumerate) {
            const auto rgb = round((static_cast<float>(in) + 1.f) * 127.5f);
            out_img[idx * comps + ch] = static_cast<uint8_t>(clamp(rgb, .0f, 255.f));
        }
    }

    stbi_write_png(filename.c_str(), w, h, comps, out_img.data(), 0);
}


template<typename T>
static void ConvertToImgAndSave(const Ort::Value& img, const string& filename) {
    const auto shape = img.GetTensorTypeAndShapeInfo().GetShape();
    ConvertToImgAndSave(TensorToSpan<T>(img, 0), static_cast<int>(shape[3]), static_cast<int>(shape[2]), static_cast<int>(shape[1]), filename);
}


template<typename T>
static Ort::Value LoadImgAndConvertToTensor(const string& filename, int comps, const Ort::AllocatorWithDefaultOptions& allocator) {
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t* img = stbi_load(filename.c_str(), &width, &height, &channels, comps);

    const vector<int64_t> shape { 1, comps, height, width };
    auto tensor = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), GetONNXType<T>());
    auto data = tensor.GetTensorMutableData<T>();
    const auto ch_size = width * height;

    for (const auto [idx, in] : span(img, ch_size * comps) | views::chunk(comps) | views::enumerate) {
        for (int64_t ch = 0; ch < comps; ++ch)
            data[ch * ch_size + idx] = static_cast<T>((in[ch] / 127.f) - 1.f);
    }

    stbi_image_free(img);
    return tensor;
}


template<typename T>
static void DumpTensor(ostream& s, const Ort::Value& tensor) {
    for (const T l1 : TensorToSpan<T>(tensor)) {
        s << static_cast<float>(l1) << ' ';
    }
    s << "\n\n";
}


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
    options.trt_force_sequential_engine_build = 1;
    // below options are strongly recommended !
    options.trt_engine_cache_enable = 1;
    options.trt_engine_cache_path = "I:/tensorrt_cache";
    session_options.AppendExecutionProvider_TensorRT(options);
}


struct Model {
    void Load(const Ort::Env& env, const wstring& model_path, bool cuda = true) {
        for (int i = 1; i < 3;++i) {
            session_options = Ort::SessionOptions();
            session_options.RegisterCustomOpsLibrary(L"ortextensions.dll", Ort::CustomOpConfigs());
            session_options.SetIntraOpNumThreads(4);

            if (i == 0) {
                AddTensorRT(session_options);
                session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions());
            }
            if (i == 1)  session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions());
            try {
                session = Ort::Session(env, model_path.c_str(), session_options);
                println("Using: {}", i);
                break;
            }
            catch (const exception& e) {
                println("Failed to create session: {}", e.what());
            }
        }

        bindings = Ort::IoBinding(session);

        wcout << model_path << '\n';
        println("Inputs:");

        input_names.resize(session.GetInputCount());
        for (auto [i, name] : input_names | views::enumerate) {
            input_types.emplace_back(session.GetInputTypeInfo(i));
            input_name_ptrs.emplace_back(session.GetInputNameAllocated(i, allocator));
            name = input_name_ptrs.back().get();

            const auto shape = input_types.back().GetTensorTypeAndShapeInfo();
            println("\t{}: {} ({})", name, (int)shape.GetElementType(), shape.GetShape());
        }

        println("Outputs:");
        const auto output_count = session.GetOutputCount();
        for (size_t i = 0; i < output_count; ++i) {
            const auto output_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            const auto name = session.GetOutputNameAllocated(i, allocator);
            bindings.BindOutput(name.get(), output_mem_info);

            const auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            println("\t{}: {} ({})", name.get(), (int)shape.GetElementType(), shape.GetShape());
        }
        println();
    }

    vector<Ort::Value> Run(span<const char*> input_names, const vector<const Ort::Value*>& input_tensors, bool recreate_outputs = false) {
        for (const auto& [name, value] : views::zip(input_names, input_tensors)) {
            bindings.BindInput(name, *value);
        }

        if (recreate_outputs) {
            bindings.ClearBoundOutputs();
            const auto output_count = session.GetOutputCount();
            for (size_t i = 0; i < output_count; ++i) {
                auto output_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                const auto name = session.GetOutputNameAllocated(i, allocator);
                bindings.BindOutput(name.get(), output_mem_info);
            }
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

    vector<Ort::Value> output_values;
};


template<typename T>
static void PadTokens(span<const T> input, span<int32_t> padded) {
    for (const auto& [converted, token] : views::zip(padded, input))
        converted = static_cast<int32_t>(token);

    ranges::fill(padded.subspan(input.size()), 49407);
}


static Ort::Value TokenizeAndPadPrompt(const string& prompt, const string& neg_prompt, int64_t padded_size, Model& tokenizer) {
    vector<const char*> prompts { neg_prompt.c_str(), prompt.c_str() };
    vector<int64_t> input_shape { ssize(prompts) };

    auto input_tensor = Ort::Value::CreateTensor(tokenizer.allocator, input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    input_tensor.FillStringTensor(prompts.data(), prompts.size());

    const auto tokenizer_output = tokenizer.Run(tokenizer.input_names, { &input_tensor }, true);

    const vector<int64_t> prompt_shape { 2, padded_size };
    auto prompt_tensor = Ort::Value::CreateTensor(tokenizer.allocator, prompt_shape.data(), prompt_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

    PadTokens(TensorToSpan<int64_t>(tokenizer_output[0], 0), TensorToSpan<int32_t>(prompt_tensor, 0));
    PadTokens(TensorToSpan<int64_t>(tokenizer_output[0], 1), TensorToSpan<int32_t>(prompt_tensor, 1));

    return prompt_tensor;
}


template<typename T>
static Ort::Value CreateLatents(Ort::AllocatorWithDefaultOptions& allocator, Model& vae_encoder) {
    std::vector<int64_t> latent_shape = { 2, 4, image_h / 8, image_w / 8 };
    Ort::Value latent = Ort::Value::CreateTensor(allocator, latent_shape.data(), latent_shape.size(), GetONNXType<T>());
    auto latent_data = TensorToSpan<T>(latent, 0);
    if (true) {
        RandomizeData<T>(latent_data);
    }
    else {
        Ort::AllocatorWithDefaultOptions tmp_allocator;
        const auto image = LoadImgAndConvertToTensor<T>("input_512.png", 3, tmp_allocator);
        auto in_latent = vae_encoder.Run(vae_encoder.input_names, { &image });
        auto in_data = TensorToSpan<T>(in_latent[0]);
        for (auto& l : in_data)
            l = static_cast<T>(static_cast<float>(l) * 0.18215f);

        ranges::copy(in_data, latent_data.data());
    }
    return latent;
}


template<typename T>
class StaticDiffusion1Pipeline {
public:
    StaticDiffusion1Pipeline(const Ort::Env& environment) : env(&environment) {}

    void LoadModels(const wstring& root) {
        tokenizer.Load(*env, L"I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/tokenizer/model.onnx");
        text_encoder.Load(*env, root + L"text_encoder/model.onnx");
        unet.Load(*env, root + L"unet/model.onnx");
        vae_encoder.Load(*env, root + L"vae_encoder/model.onnx");
        vae_decoder.Load(*env, root + L"vae_decoder/model.onnx");
    }

    void Run(const string& pos_prompt, const string& neg_prompt, size_t steps, float cfg, int image_count = 1, optional<uint32_t> seed = {}) {
        const auto start_time = chrono::high_resolution_clock::now();

        const auto prompt_tokens = TokenizeAndPadPrompt(pos_prompt, neg_prompt, 77, tokenizer);
        const auto embeddings = text_encoder.Run(text_encoder.input_names, { &prompt_tokens });

        auto latent = CreateLatents<T>(unet.allocator, vae_encoder);
        const auto latents = TensorToSpan<T>(latent, 0);

        std::vector<int64_t> ts_shape = { 1 };
        auto timestep_tensor = Ort::Value::CreateTensor(unet.allocator, ts_shape.data(), ts_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        auto timestep = TensorToSpan<int64_t>(timestep_tensor);

        PNDMScheduler<T> scheduler(1000, 0.00085f, 0.012f, "scaled_linear", {}, true, false, "epsilon", 1);

        scheduler.set_timesteps(40);
        auto timesteps = scheduler.timesteps();

        /*
            const float strength = .5f;
            timesteps = vector(timesteps.begin() + timesteps.size() * (1.f - strength), timesteps.end());

            vector<T> noise(latents.size());
            RandomizeData<T>(noise);
            scheduler.add_noise_to_sample(latents, noise, timesteps[0]);
        */

        println("Preparation time: {}", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time));

        for (int i = 0; i < image_count; ++i) {
            const auto img_start_time = chrono::high_resolution_clock::now();

            RandomizeData<T>(latents, seed);

            for (const auto t : timesteps) {
                timestep[0] = t;
                scheduler.scale_model_input(latents, t);

                ranges::copy(TensorToSpan<T>(latent, 0), TensorToSpan<T>(latent, 1).data());

                auto noise_predictions = unet.Run(unet.input_names, { &latent, &timestep_tensor, &embeddings[0] });
                const auto pred_uncond = TensorToSpan<T>(noise_predictions[0], 0);
                auto pred_cond = TensorToSpan<T>(noise_predictions[0], 1);

                for (auto [np_cond, np_uncond] : views::zip(pred_cond, pred_uncond))
                    np_cond = static_cast<T>(static_cast<float>(np_uncond) + cfg * (static_cast<float>(np_cond) - static_cast<float>(np_uncond)));

                scheduler.step(pred_cond, t, latents);
            }

            for (auto& v : latents) v = static_cast<T>(static_cast<float>(v) * (1.f / 0.18215f));
            const auto img = vae_decoder.Run(vae_decoder.input_names, { &latent });

            println("Image generated in {}", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - img_start_time));

            ConvertToImgAndSave<T>(img[0], format("out_{:03}_steps{}_cfg{:02.1f}.png", i, timesteps.size(), cfg));
        }
    }

private:
    const Ort::Env* env;
    Model tokenizer;
    Model text_encoder;
    Model unet;
    Model vae_encoder;
    Model vae_decoder;
};


template<typename T>
class StaticDiffusionXLPipeline {
public:
    StaticDiffusionXLPipeline(const Ort::Env& environment) : env(&environment) {}

    void LoadModels(const wstring& root) {
        tokenizer_1.Load(*env, root + L"tokenizer/model.onnx");
        tokenizer_2.Load(*env, root + L"tokenizer_2/model.onnx");
        text_encoder_1.Load(*env, root + L"text_encoder/model.onnx");
        text_encoder_2.Load(*env, root + L"text_encoder_2/model.onnx");
        unet.Load(*env, root + L"unet/model.onnx");
        vae_encoder.Load(*env, root + L"vae_encoder/model.onnx");
        vae_decoder.Load(*env, root + L"vae_decoder/model.onnx");
    }

    void Run(const string& pos_prompt, const string& neg_prompt, size_t steps, float cfg, int image_count = 1, optional<uint32_t> seed = {}) {

    }

private:
    const Ort::Env* env;
    Model tokenizer_1;
    Model tokenizer_2;
    Model text_encoder_1;
    Model text_encoder_2;
    Model unet;
    Model vae_encoder;
    Model vae_decoder;
};


int main(int argc, char* argv[]) {
    try {
        Ort::Env env { ORT_LOGGING_LEVEL_WARNING, "" };
        println("Available providers: {}", Ort::GetAvailableProviders());

        const auto start_time = chrono::high_resolution_clock::now();

        StaticDiffusion1Pipeline<Ort::Float16_t> pipeline(env);
        pipeline.LoadModels(L"I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/");
 
       // StaticDiffusionXLPipeline<Ort::Float16_t> pipeline(env);
       // pipeline.LoadModels(L"I:/huggingface/hub/models--onnxruntime--sdxl-turbo/snapshots/bd6180e5aa5a5e326fbb0ba1bdda15cb3817f63c/");

        println("Models loaded in {}", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time));

        pipeline.Run("Woman, red hair, fit. Photo",
                     //"blurry, render, drawing, painting, art, cgi, deformed, wrong, weird",
                     "",
                     40, 5.5f, 10);
    }
    catch (const Ort::Exception& e) {
        println("Exception ({}): {}", (int)e.GetOrtErrorCode(), e.what());
        return -1;
    }
    catch (const exception& e) {
        println("Exception: {}",  e.what());
        return -1;
    }

    return 0;
}
