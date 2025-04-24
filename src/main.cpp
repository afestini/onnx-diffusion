#include <assert.h>
#include <iostream>
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


template<typename T>
static span<const T> TensorToSpan(const Ort::Value& tensor) {
    return span(tensor.GetTensorData<T>(), tensor.GetTensorTypeAndShapeInfo().GetElementCount());
}

template<typename T>
static span<T> TensorToSpan(Ort::Value& tensor) {
    return span(tensor.GetTensorMutableData<T>(), tensor.GetTensorTypeAndShapeInfo().GetElementCount());
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


static void RandomizeTensor(Ort::Value& tensor) {
    std::random_device rd;
    //std::mt19937 gen(rd());
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const auto size = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    auto data = span(tensor.GetTensorMutableData<Ort::Float16_t>(), size);

    for (auto& val : data) val = static_cast<Ort::Float16_t>(dist(gen));
}


static void ConvertToImgAndSave(const Ort::Value& img, const string& filename) {
    const auto data = img.GetTensorData<Ort::Float16_t>();
    const auto shape = img.GetTensorTypeAndShapeInfo().GetShape();
    const auto size = img.GetTensorTypeAndShapeInfo().GetElementCount();
    const auto ch_size = shape[2] * shape[3];

    vector<uint8_t> out_img(size);
    for (const auto [ch, channel_pixels] : span(data, size) | views::chunk(ch_size) | views::enumerate) {
        for (const auto [idx, in] : channel_pixels | views::enumerate)
            out_img[idx * shape[1] + ch] = static_cast<uint8_t>((static_cast<float>(in) + 1.f) * 127.5f);
    }

    stbi_write_png(filename.c_str(), static_cast<int>(shape[3]), static_cast<int>(shape[2]), static_cast<int>(shape[1]), out_img.data(), 0);
}


static void ConvertToImgAndSave(span<const float> img, int w, int h, int comps, const string& filename) {
    const auto ch_size = w * h;

    vector<uint8_t> out_img(img.size());

    for (const auto [ch, channel_pixels] : img | views::chunk(ch_size) | views::enumerate) {
        for (const auto [idx, in] : channel_pixels | views::enumerate)
            out_img[idx * comps + ch] = static_cast<uint8_t>((static_cast<float>(in) + 1.f) * 127.5f);
    }

    stbi_write_png(filename.c_str(), w, h, comps, out_img.data(), 0);
}


static Ort::Value LoadImgAndConvertToTensor(const string& filename, int comps, const Ort::AllocatorWithDefaultOptions& allocator) {
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t* img = stbi_load(filename.c_str(), &width, &height, &channels, comps);

    const vector<int64_t> shape { 1, comps, height, width };
    auto tensor = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    auto data = tensor.GetTensorMutableData<Ort::Float16_t>();
    const auto ch_size = width * height;

    for (const auto [idx, in] : span(img, width * height * comps) | views::chunk(comps) | views::enumerate) {
        for (int64_t ch = 0; ch < comps; ++ch)
            data[ch * ch_size + idx] = static_cast<Ort::Float16_t>((in[ch] / 127.f) - 1.f);
    }

    stbi_image_free(img);
    return tensor;
}


static void DumpTensor(ostream& s, const Ort::Value& tensor) {
    for (const Ort::Float16_t l1 : TensorToSpan<Ort::Float16_t>(tensor)) {
        s << l1.ToFloat() << ' ';
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
            //Ort::MemoryInfo output_mem_info(name, OrtDeviceAllocator, 0, OrtMemTypeDefault);
            const auto output_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            const auto name = session.GetOutputNameAllocated(i, allocator);
            bindings.BindOutput(name.get(), output_mem_info);

            const auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            println("\t{}: {} ({})", name.get(), (int)shape.GetElementType(), shape.GetShape());
        }
        println();
    }

    vector<Ort::Value> Run(span<const char*> input_names, const vector<const Ort::Value*>& input_tensors, bool recreate_outputs = false) {
        bindings.ClearBoundInputs();
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


static Ort::Value TokenizeAndPadPrompt(const string& prompt, size_t padded_size, Model& tokenizer) {
    vector<const char*> prompts { prompt.c_str() };
    vector<int64_t> input_shape { ssize(prompts) };

    auto input_tensor = Ort::Value::CreateTensor(tokenizer.allocator, input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    input_tensor.FillStringTensor(prompts.data(), prompts.size());
    const auto wtf = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

    const auto tokenizer_output = tokenizer.Run(tokenizer.input_names, { &input_tensor }, true);
    const auto tokens = TensorToSpan<int64_t>(tokenizer_output[0]);

    const vector<int64_t> prompt_shape { 1, static_cast<int64_t>(max(padded_size, tokens.size())) };
    auto prompt_tensor = Ort::Value::CreateTensor(tokenizer.allocator, prompt_shape.data(), prompt_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    auto conv_tokens = TensorToSpan<int32_t>(prompt_tensor);

    for (const auto& [token, converted] : views::zip(tokens, conv_tokens))
        converted = static_cast<int32_t>(token);

    ranges::fill(conv_tokens.subspan(tokens.size()), 49407);

    return prompt_tensor;
}


static Ort::Value EncodePromptTokens(Model& text_encoder, const Ort::Value& tokens) {
    const auto uncond_encoded = text_encoder.Run(text_encoder.input_names, { &tokens });
    return CopyTensor<Ort::Float16_t>(uncond_encoded[0], text_encoder.allocator);
}


static Ort::Value CreateLatent(Ort::AllocatorWithDefaultOptions& allocator, Model& vae_encoder) {
    std::vector<int64_t> latent_shape = { 1, 4, 64, 64 };
    Ort::Value latent = Ort::Value::CreateTensor(allocator, latent_shape.data(), latent_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    if (true) {
        RandomizeTensor(latent);
    }
    else {
        Ort::AllocatorWithDefaultOptions tmp_allocator;
        const auto latent_in = LoadImgAndConvertToTensor("input_512.png", 3, tmp_allocator);
        ConvertToImgAndSave(latent_in, "in_sanity.png");
        latent = move(vae_encoder.Run(vae_encoder.input_names, { &latent_in })[0]);

        //for (auto& l : TensorToSpan<Ort::Float16_t>(latent))
        //    l = Ort::Float16_t(l.ToFloat() * 0.18215f);
    }
    return latent;
}


static Ort::Value RunPrediction(Model& unet, const Ort::Value& latent, const Ort::Value& timestep, const Ort::Value& text_embeddings) {
    const auto prediction = unet.Run(unet.input_names, { &latent, &timestep, &text_embeddings });
    return CopyTensor<Ort::Float16_t>(prediction[0], unet.allocator);
}


static int run_inference(Model& tokenizer, Model& text_encoder, Model& unet, Model& vae_dec, Model& vae_enc) {
    const auto uncond_prompt = TokenizeAndPadPrompt("", 77, tokenizer);
    const auto uncond_encoder_output = EncodePromptTokens(text_encoder, uncond_prompt);

    const auto cond_prompt = TokenizeAndPadPrompt("Dog", 77, tokenizer);
    const auto cond_encoder_output = EncodePromptTokens(text_encoder, cond_prompt);

    auto latent = CreateLatent(unet.allocator, vae_enc);
    const auto latents = TensorToSpan<Ort::Float16_t>(latent);

    std::vector<int64_t> ts_shape = { 1 };
    auto timestep_tensor = Ort::Value::CreateTensor(unet.allocator, ts_shape.data(), ts_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto& timestep = timestep_tensor.GetTensorMutableData<int64_t>()[0];

    PNDMScheduler scheduler(1000, 0.00085f, 0.012f, "scaled_linear", {}, true, false, "epsilon", 1);

    scheduler.set_timesteps(30);
    const auto timesteps = scheduler.timesteps();
    const float guidance_scale = 7.f;

    for (const auto t : scheduler.timesteps()) {
        timestep = t;

        scheduler.scale_model_input(latents, t);

        auto noise_pred_cond = RunPrediction(unet, latent, timestep_tensor, cond_encoder_output);
        auto noise_cond = TensorToSpan<Ort::Float16_t>(noise_pred_cond);

        const auto noise_pred_uncond = RunPrediction(unet, latent, timestep_tensor, uncond_encoder_output);
        const auto noise_uncond = TensorToSpan<Ort::Float16_t>(noise_pred_uncond);

        for (auto [np_cond, np_uncond] : views::zip(noise_cond, noise_uncond))
            np_cond = Ort::Float16_t(np_uncond.ToFloat() + guidance_scale * (np_cond.ToFloat() - np_uncond.ToFloat()));

        scheduler.step(noise_cond, t, latents);

        //auto tmp = CopyTensor<Ort::Float16_t>(latent, unet.allocator);
        //for (auto& v : TensorToSpan<Ort::Float16_t>(tmp)) v = Ort::Float16_t(v.ToFloat() / 0.18215f);
        //const auto img = vae_dec.Run(vae_dec.input_names, { &tmp });
        //ConvertToImgAndSave(img[0], format("out_{}.png", t));
    }

    for (auto& v : latents) v = Ort::Float16_t(v.ToFloat() / 0.18215f);
    const auto img = vae_dec.Run(vae_dec.input_names, { &latent });
    ConvertToImgAndSave(img[0], "out.png");

    return 0;
}


int main(int argc, char* argv[]) {
    try {
        println("Available providers: {}", Ort::GetAvailableProviders());

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "");

        //const wstring root_dir = L"I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/";
        const wstring root_dir = L"I:/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/";
        Model tokenizer(env,    root_dir + L"tokenizer/model.onnx");
        Model text_encoder(env, root_dir + L"text_encoder/model.onnx");
        Model unet(env,         root_dir + L"unet/model.onnx");
        Model vae_encoder(env,  root_dir + L"vae_encoder/model.onnx");
        Model vae_decoder(env,  root_dir + L"vae_decoder/model.onnx");

        run_inference(tokenizer, text_encoder, unet, vae_decoder, vae_encoder);
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
