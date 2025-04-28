#include <assert.h>
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

static constexpr int64_t image_dim = 512;


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
static void RandomizeData(span<T> data) {
    std::random_device rd;
    std::mt19937 gen(rd());
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


struct Model {
    Model(const Ort::Env& env, const wstring& model_path, bool cuda = true) {
        session_options.RegisterCustomOpsLibrary(L"ortextensions.dll", Ort::CustomOpConfigs());
        session_options.SetIntraOpNumThreads(4);

        if (cuda)
            session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions());

        session = Ort::Session(env, model_path.c_str(), session_options);
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


static Ort::Value TokenizeAndPadPrompt(const string& prompt, int64_t padded_size, Model& tokenizer) {
    vector<const char*> prompts { "", prompt.c_str() };
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
    std::vector<int64_t> latent_shape = { 2, 4, image_dim / 8, image_dim / 8 };
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
static void run_inference(Model& tokenizer, Model& text_encoder, Model& unet, Model& vae_dec, Model& vae_enc) {
    const auto prompt_tokens = TokenizeAndPadPrompt("Woman. Photo.", 77, tokenizer);
    const auto embeddings = text_encoder.Run(text_encoder.input_names, { &prompt_tokens });

    auto latent = CreateLatents<T>(unet.allocator, vae_enc);
    const auto latents = TensorToSpan<T>(latent, 0);

    std::vector<int64_t> ts_shape = { 1 };
    auto timestep_tensor = Ort::Value::CreateTensor(unet.allocator, ts_shape.data(), ts_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto& timestep = timestep_tensor.GetTensorMutableData<int64_t>()[0];

    PNDMScheduler<T> scheduler(1000, 0.00085f, 0.012f, "scaled_linear", {}, true, false, "epsilon", 1);
    //USTMScheduler<T> scheduler(1000, 0.00085f, 0.012f, "scaled_linear", {}, false, "epsilon", 1);
    //EulerAScheduler<T> scheduler;

    scheduler.set_timesteps(20);
    auto timesteps = scheduler.timesteps();
    const float guidance_scale = 7.f;

/*
    const float strength = .5f;
    timesteps = vector(timesteps.begin() + timesteps.size() * (1.f - strength), timesteps.end());

    vector<T> noise(latents.size());
    RandomizeData<T>(noise);
    scheduler.add_noise_to_sample(latents, noise, timesteps[0]);
*/

    for (const auto t : timesteps) {
        timestep = t;

        scheduler.scale_model_input(latents, t);

        ranges::copy(TensorToSpan<T>(latent, 0), TensorToSpan<T>(latent, 1).data());

        auto noise_predictions = unet.Run(unet.input_names, { &latent, &timestep_tensor, &embeddings[0] });
        const auto pred_uncond = TensorToSpan<T>(noise_predictions[0], 0);
        auto pred_cond = TensorToSpan<T>(noise_predictions[0], 1);

        for (auto [np_cond, np_uncond] : views::zip(pred_cond, pred_uncond))
            np_cond = static_cast<T>(static_cast<float>(np_uncond) + guidance_scale * (static_cast<float>(np_cond) - static_cast<float>(np_uncond)));

        scheduler.step(pred_cond, t, latents);
    }

    for (auto& v : latents) v = static_cast<T>(static_cast<float>(v) * (1.f / 0.18215f));
    const auto img = vae_dec.Run(vae_dec.input_names, { &latent });
    ConvertToImgAndSave<T>(img[0], "out.png");

    return 0;
}


int main(int argc, char* argv[]) {
    try {
        println("Available providers: {}", Ort::GetAvailableProviders());

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "");

        const wstring root_dir = L"I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/";
        //const wstring root_dir = L"I:/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/da5601014c467b382fcf42019ba920a903f2103e/";
        Model tokenizer(env,    L"I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/tokenizer/model.onnx");
        Model text_encoder(env, root_dir + L"text_encoder/model.onnx");
        Model unet(env,         root_dir + L"unet/model.onnx");
        Model vae_encoder(env,  root_dir + L"vae_encoder/model.onnx");
        Model vae_decoder(env,  root_dir + L"vae_decoder/model.onnx");

        run_inference<Ort::Float16_t>(tokenizer, text_encoder, unet, vae_decoder, vae_encoder);
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
