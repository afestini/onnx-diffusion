#include <iostream>
#include <filesystem>
#include <memory>
#include <numeric>
#include <print>
#include <ranges>

#include <onnxruntime_cxx_api.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "pipeline.h"
#include "scheduler.h"
#include "json.h"

using namespace std;


static constexpr int64_t image_w = 512;
static constexpr int64_t image_h = 512;


template<typename T>
static consteval ONNXTensorElementDataType GetONNXType() {
    if constexpr (is_same_v<T, float>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if constexpr (is_same_v<T, Ort::Float16_t>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if constexpr (is_same_v<T, int32_t>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    if constexpr (is_same_v<T, int64_t>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
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


template<typename OldType, typename NewType>
static Ort::Value ConvertTensor(const Ort::Value& tensor, const Ort::AllocatorWithDefaultOptions& allocator) {
    const auto info = tensor.GetTensorTypeAndShapeInfo();
    auto new_tensor = Ort::Value::CreateTensor(allocator,
                                               info.GetShape().data(),
                                               info.GetShape().size(),
                                               GetONNXType<NewType>());
    for (const auto& [in, out] : views::zip(TensorToSpan<OldType>(tensor), TensorToSpan<NewType>(new_tensor)))
        out = static_cast<NewType>(in);

    return new_tensor;
}


static void RandomizeData(span<float> data, float scale, optional<uint32_t> seed = {}) {
    random_device rd;
    mt19937 gen(seed ? *seed : rd());
    normal_distribution<float> dist(0.f, 1.f);

    for (auto& val : data) val = dist(gen) * scale;
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

    for (const auto [idx, in] : span(img, ch_size* comps) | views::chunk(comps) | views::enumerate) {
        for (int64_t ch = 0; ch < comps; ++ch)
            data[ch * ch_size + idx] = static_cast<T>((in[ch] / 127.f) - 1.f);
    }

    stbi_image_free(img);
    return tensor;
}


template<typename T>
void LoadTensor(const string& path, span<T> tensor) {
    ifstream(path, ios::binary | ios::in).read((char*)tensor.data(), tensor.size_bytes());
}

template<typename T>
void SaveTensor(const string& path, span<T> tensor) {
    ofstream(path, ios::binary | ios::out).write((const char*)tensor.data(), tensor.size_bytes());
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
    void Load(const Ort::Env& env, const filesystem::path& model_path, bool cuda = true) {
        for (int i = 1; i < 3; ++i) {
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


template<typename PaddedType>
static void PadTokens(span<const int64_t> input, PaddedType pad_token, span<PaddedType> padded) {
    for (const auto& [converted, token] : views::zip(padded, input))
        converted = static_cast<PaddedType>(token);

    ranges::fill(padded.subspan(input.size()), pad_token);
}


template<typename PaddedType>
static Ort::Value TokenizeAndPadPrompt(const string& prompt, const string& neg_prompt, int64_t padded_size, PaddedType pad_token, Model& tokenizer) {
    vector<const char*> prompts { neg_prompt.c_str(), prompt.c_str() };
    vector<int64_t> input_shape { ssize(prompts) };

    auto input_tensor = Ort::Value::CreateTensor(tokenizer.allocator, input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    input_tensor.FillStringTensor(prompts.data(), prompts.size());

    const auto tokenizer_output = tokenizer.Run(tokenizer.input_names, { &input_tensor }, true);

    const vector<int64_t> prompt_shape { 2, padded_size };
    auto prompt_tensor = Ort::Value::CreateTensor(tokenizer.allocator, prompt_shape.data(), prompt_shape.size(), GetONNXType<PaddedType>());

    PadTokens<PaddedType>(TensorToSpan<int64_t>(tokenizer_output[0], 0), pad_token, TensorToSpan<PaddedType>(prompt_tensor, 0));
    PadTokens<PaddedType>(TensorToSpan<int64_t>(tokenizer_output[0], 1), pad_token, TensorToSpan<PaddedType>(prompt_tensor, 1));

    return prompt_tensor;
}


template<typename T>
static Ort::Value CreateLatents(Ort::AllocatorWithDefaultOptions& allocator, Model& vae_encoder, float scale) {
    vector<int64_t> latent_shape = { 2, 4, image_h / 8, image_w / 8 };
    Ort::Value latent = Ort::Value::CreateTensor(allocator, latent_shape.data(), latent_shape.size(), GetONNXType<T>());
    return latent;
}


template<typename T>
static void LoadAndEncodeImg(const string& path, Model& vae_encoder, float scale, vector<float>& data) {
    Ort::AllocatorWithDefaultOptions tmp_allocator;
    const auto image = LoadImgAndConvertToTensor<T>(path, 3, tmp_allocator);
    auto encoded = vae_encoder.Run(vae_encoder.input_names, { &image });

    const auto enc_data = TensorToSpan<T>(encoded[0]);
    data.resize(enc_data.size());
    for (const auto& [in, out] : views::zip(enc_data, data))
        out = static_cast<float>(in) * scale;
}


template<typename T>
class StableDiffusion1Pipeline : public Pipeline {
public:
    using Pipeline::Pipeline;

    void LoadModels(const filesystem::path& root) override {
        tokenizer.Load(*env, L"I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/tokenizer/model.onnx");
        text_encoder.Load(*env, root/"text_encoder/model.onnx");
        unet.Load(*env, root/"unet/model.onnx");
        vae_encoder.Load(*env, root/"vae_encoder/model.onnx");
        vae_decoder.Load(*env, root/"vae_decoder/model.onnx");
        scheduler = Scheduler::Create(root / "scheduler/scheduler_config.json");
    }

    void Run(const string& pos_prompt, const string& neg_prompt, size_t steps,
             float cfg, int image_count = 1, optional<uint32_t> seed = {},
             const string& img = "", float denoise_strength = .5f) override {
        const auto start_time = chrono::high_resolution_clock::now();

        if (!scheduler) return;

        scheduler->set_timesteps(steps);
        const auto& timesteps = scheduler->timesteps();

        // Tokenize prompts
        const auto prompt_tokens = TokenizeAndPadPrompt<int32_t>(pos_prompt, neg_prompt, 77, 49407, tokenizer);
        const auto embeddings = text_encoder.Run(text_encoder.input_names, { &prompt_tokens });

        // Prepare latent tensor
        auto unet_latent = CreateLatents<T>(unet.allocator, vae_encoder, scheduler->init_noise_sigma());
        const auto unet_latents = TensorToSpan<T>(unet_latent, 0);
        vector<float> latent(unet_latents.size());
        vector<float> noise_pred(unet_latents.size());
        vector<float> init_img;

        if (!img.empty()) LoadAndEncodeImg<T>("input_512.png", vae_encoder, 0.18215f, init_img);

        // Prepare timestep tensor
        vector<int64_t> ts_shape = { 1 };
        auto timestep_tensor = Ort::Value::CreateTensor(unet.allocator, ts_shape.data(), ts_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        auto timestep = TensorToSpan<int64_t>(timestep_tensor);

        println("Preparation time: {}", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time));

        for (int i = 0; i < image_count; ++i) {
            const auto img_start_time = chrono::high_resolution_clock::now();

            int start_timestep_idx = 0;
            if (init_img.empty()) {
                RandomizeData(latent, scheduler->init_noise_sigma(), seed);
            }
            else {
                start_timestep_idx = static_cast<int>(round((1.f - denoise_strength) * static_cast<float>(timesteps.size())));
                ranges::copy(init_img, latent.data());
                scheduler->add_noise_to_sample(latent, timesteps[start_timestep_idx]);
            }
            scheduler->reset();

            // Scheduler loop
            for (const auto t : views::drop(timesteps, start_timestep_idx)) {
                timestep[0] = t;

                const auto scale_factor = scheduler->scale_model_input_factor(t);
                for (const auto& [in, out] : views::zip(latent, unet_latents))
                    out = T(in * scale_factor);

                ranges::copy(unet_latents, TensorToSpan<T>(unet_latent, 1).data());

                auto noise_predictions = unet.Run(unet.input_names, { &unet_latent, &timestep_tensor, &embeddings[0] });
                const auto pred_uncond = TensorToSpan<T>(noise_predictions[0], 0);
                auto pred_cond = TensorToSpan<T>(noise_predictions[0], 1);

                for (auto [np, np_cond, np_uncond] : views::zip(noise_pred, pred_cond, pred_uncond))
                    np = static_cast<float>(np_uncond) + cfg * (static_cast<float>(np_cond) - static_cast<float>(np_uncond));

                scheduler->step(noise_pred, t, latent);
            }

            // Scale, decode and save image
            for (const auto& [in, out] : views::zip(latent, unet_latents)) out = static_cast<T>(in * (1.f / 0.18215f));
            const auto img = vae_decoder.Run(vae_decoder.input_names, { &unet_latent });

            println("Image generated in {}", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - img_start_time));

            ConvertToImgAndSave<T>(img[0], format("out_{:03}_steps{}_cfg{:02.1f}.png", i, timesteps.size(), cfg));
        }
    }

private:
    Model tokenizer;
    Model text_encoder;
    Model unet;
    Model vae_encoder;
    Model vae_decoder;
};


template<typename T>
class StableDiffusionXLPipeline : public Pipeline {
public:
    using Pipeline::Pipeline;

    void LoadModels(const filesystem::path& root) override {
        tokenizer_1.Load(*env, root/"tokenizer/model.onnx");
        tokenizer_2.Load(*env, root/"tokenizer_2/model.onnx");
        text_encoder_1.Load(*env, root/"text_encoder/model.onnx");
        text_encoder_2.Load(*env, root/"text_encoder_2/model.onnx");
        unet.Load(*env, root/"unet/model.onnx");
        vae_encoder.Load(*env, root/"vae_encoder/model.onnx");
        vae_decoder.Load(*env, root/"vae_decoder/model.onnx");
        scheduler = Scheduler::Create(root / "scheduler/scheduler_config.json");
    }

    void Run(const string& pos_prompt, const string& neg_prompt, size_t steps,
             float cfg, int image_count = 1, optional<uint32_t> seed = {},
             const string& img = "", float denoise_strength = .5f) override {
        const auto start_time = chrono::high_resolution_clock::now();

        if (!scheduler) return;

        scheduler->set_timesteps(steps);
        const auto& timesteps = scheduler->timesteps();

        // Tokenize prompts
        const auto prompt_tokens_1 = TokenizeAndPadPrompt<int32_t>(pos_prompt, neg_prompt, 77, 49407, tokenizer_1);
        const auto prompt_tokens_2 = TokenizeAndPadPrompt<int64_t>(pos_prompt, neg_prompt, 77, 0, tokenizer_2);
        const auto embeddings_1 = text_encoder_1.Run(text_encoder_1.input_names, { &prompt_tokens_1 });
        const auto embeddings_2 = text_encoder_2.Run(text_encoder_2.input_names, { &prompt_tokens_2 });

        // Concat prompt embeddings
        vector<int64_t> concat_shape = { 2, 77, 2048 };
        auto embeddings_concat = Ort::Value::CreateTensor(unet.allocator, concat_shape.data(), concat_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        const auto emb1 = TensorToSpan<Ort::Float16_t>(embeddings_1[0]);
        const auto emb2 = TensorToSpan<Ort::Float16_t>(embeddings_2[1]);
        auto out = embeddings_concat.GetTensorMutableData<Ort::Float16_t>();
        for (const auto& [in1, in2] : views::zip(views::chunk(emb1, 768), views::chunk(emb2, 1280))) {
            for (Ort::Float16_t v : in1) *out++ = v;
            for (Ort::Float16_t v : in2) *out++ = v;
        }

        // Prepare latent tensor
        auto unet_latent = CreateLatents<T>(unet.allocator, vae_encoder, scheduler->init_noise_sigma());
        const auto unet_latents = TensorToSpan<T>(unet_latent, 0);
        vector<float> latent(unet_latents.size());
        vector<float> noise_pred(unet_latents.size());
        vector<float> init_img;

        if (!img.empty()) LoadAndEncodeImg<T>("input_512.png", vae_encoder, 0.18215f, init_img);

        // Prepare timestep tensor
        vector<int64_t> ts_shape = { 1 };
        auto timestep_tensor = Ort::Value::CreateTensor(unet.allocator, ts_shape.data(), ts_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        auto timestep = TensorToSpan<Ort::Float16_t>(timestep_tensor);

        // Prepare time ids tensor
        vector<int64_t> ti_shape = { 2, 6 };
        auto time_ids = Ort::Value::CreateTensor(unet.allocator, ti_shape.data(), ti_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        auto tids = TensorToSpan<Ort::Float16_t>(time_ids, 0);
        tids[0] = Ort::Float16_t(512.f);
        tids[1] = Ort::Float16_t(512.f);
        tids[2] = Ort::Float16_t(0.f);
        tids[3] = Ort::Float16_t(0.f);
        tids[4] = Ort::Float16_t(512.f);
        tids[5] = Ort::Float16_t(512.f);
        ranges::copy(TensorToSpan<T>(time_ids, 0), TensorToSpan<T>(time_ids, 1).data());

        println("Preparation time: {}", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time));

        for (int i = 0; i < image_count; ++i) {
            const auto img_start_time = chrono::high_resolution_clock::now();

            int start_timestep_idx = 0;
            if (init_img.empty()) {
                RandomizeData(latent, scheduler->init_noise_sigma(), seed);
            }
            else {
                start_timestep_idx = static_cast<int>(round((1.f - denoise_strength) * static_cast<float>(timesteps.size())));
                ranges::copy(init_img, latent.data());
                scheduler->add_noise_to_sample(latent, timesteps[start_timestep_idx]);
            }
            scheduler->reset();

            // Scheduler loop
            for (const auto t : views::drop(timesteps, start_timestep_idx)) {
                timestep[0] = Ort::Float16_t(float(t));

                const auto scale_factor = scheduler->scale_model_input_factor(t);
                for (const auto& [in, out] : views::zip(latent, unet_latents))
                    out = T(in * scale_factor);

                ranges::copy(unet_latents, TensorToSpan<T>(unet_latent, 1).data());

                auto noise_predictions = unet.Run(unet.input_names, { &unet_latent, &timestep_tensor, &embeddings_concat, &embeddings_2[0], &time_ids });
                const auto pred_uncond = TensorToSpan<T>(noise_predictions[0], 0);
                auto pred_cond = TensorToSpan<T>(noise_predictions[0], 1);

                for (auto [np, np_cond, np_uncond] : views::zip(noise_pred, pred_cond, pred_uncond))
                    np = static_cast<float>(np_uncond) + cfg * (static_cast<float>(np_cond) - static_cast<float>(np_uncond));

                scheduler->step(noise_pred, t, latent);
            }

            // Scale, decode and save image
            for (const auto& [in, out] : views::zip(latent, unet_latents)) out = static_cast<T>(in * (1.f / 0.13025f));
            const auto img = vae_decoder.Run(vae_decoder.input_names, { &unet_latent });

            println("Image generated in {}", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - img_start_time));

            ConvertToImgAndSave<T>(img[0], format("out_{:03}_steps{}_cfg{:02.1f}.png", i, timesteps.size(), cfg));
        }
    }

private:
    Model tokenizer_1;
    Model tokenizer_2;
    Model text_encoder_1;
    Model text_encoder_2;
    Model unet;
    Model vae_encoder;
    Model vae_decoder;
};




shared_ptr<Pipeline> Pipeline::Load(const Ort::Env& env, const string& root) {
    JsonParser parser;
    const auto json = parser.Parse(filesystem::path(root)/"model_index.json");
    const auto pipeline_class = json["_class_name"].as<string>();

    shared_ptr<Pipeline> pipeline;
    if (pipeline_class.ends_with("StableDiffusionXLPipeline"))
        pipeline = make_shared<StableDiffusionXLPipeline<Ort::Float16_t>>(env);
    else if (pipeline_class.ends_with("StableDiffusionPipeline"))
        pipeline = make_shared<StableDiffusion1Pipeline<Ort::Float16_t>>(env);
    else
        return {};

    pipeline->LoadModels(root);
    return pipeline;
}


void Pipeline::LoadDefaultScheduler() {}
