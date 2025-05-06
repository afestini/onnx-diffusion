#include <chrono>
#include <fstream>
#include <print>
#include <ranges>
#include <span>

#include <onnxruntime_cxx_api.h>

#include "pipeline.h"
#include "scheduler.h"

using namespace std;


string read_file(const string& filepath) {
    auto file = fopen(filepath.c_str(), "rb");
    if (!file) throw runtime_error("Could not open file: " + filepath);

    fpos_t len {};
    fseek(file, 0, SEEK_END);
    fgetpos(file, &len);
    fseek(file, 0, SEEK_SET);

    string data;
    data.resize(len);
    fread(data.data(), 1, data.size(), file);
    fclose(file);

    return data;
}


#include "clip_tokenizer.h"



int main() {
    try {
        string vocab_file = "I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/tokenizer/vocab.json";
        string merges_file = "I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/tokenizer/merges.txt";

        // Benchmark Tokenizer Initialization
        const auto start_init = chrono::high_resolution_clock::now();
        ClipTokenizer tokenizer(vocab_file, merges_file);
        //AsciiClipTokenizer tokenizer(vocab_file, merges_file);
        const auto end_init = chrono::high_resolution_clock::now();
        const auto init_duration = chrono::duration_cast<chrono::milliseconds>(end_init - start_init);

        //println("Tokenizer initialized. Vocab size: {}, Merges: {}", tokenizer.vocab_to_id_.size(), tokenizer.merge_ranks_.size());
        println("Tokenizer initialization time: {}", init_duration);

        const string text = read_file("test.txt");

        // Benchmark Encoding
        const auto start_encode = chrono::high_resolution_clock::now();
/*
        size_t total_len = 0;
        for (int i = 0; i < 10000; ++i)
            total_len += tokenizer.encode(text, 77).size();

        println("{}", total_len);
*/
        const auto token_ids = tokenizer.encode(text, 77);

        const auto end_encode = chrono::high_resolution_clock::now();
        const auto encode_duration = chrono::duration_cast<chrono::microseconds>(end_encode - start_encode);

        println("Encoded text: {}", token_ids);
        println("Encoding time: {}", encode_duration);

        // Benchmark Decoding
        const auto start_decode = chrono::high_resolution_clock::now();
        const string decoded_text = tokenizer.decode(token_ids);
        const auto end_decode = chrono::high_resolution_clock::now();
        const auto decode_duration = chrono::duration_cast<chrono::microseconds>(end_decode - start_decode);

        println("Decoded text: {}", decoded_text);
        println("Decoding time: {}", decode_duration);
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}








#if 0

int main(int argc, char* argv[]) {
    const auto merges_file = "I:/huggingface/hub/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304/tokenizer/merges.txt";
    ifstream file(merges_file, ios::ate);

    if (!file)
        return -1;

    string test = "happy dire wolves go bananas for pineapple";
    auto groups = views::transform(test, [](auto& c) {return string_view(&c, 1); }) | ranges::to<vector>();

    const auto sz = file.tellg();
    file.seekg(0, ios::beg);

    vector<char> merges(sz);
    file.read(merges.data(), sz);

    const auto tok_start = chrono::steady_clock::now();

    for (auto line : views::split(merges, '\n') | views::drop(1)) {
        for (auto [a, b] : views::split(line, ' ') | views::transform([](const auto& e) { return string_view(e); }) | views::adjacent<2>) {
            //println("{} - {}", a, b);
            //for (const auto& [l, r] : views::filter(groups, [](const auto sv) {return !sv.empty(); }) | views::adjacent<2>) {
            //    if (l == a && r == b) {
            //        l = string_view(l.data(), l.size() + r.size());
            //        r = {};
            //    }
            //}
        }
    }

    println("{}", chrono::steady_clock::now() - tok_start);


/*
    
    while (true) {
        for (const auto& [l, r] : views::filter(groups, [](const auto sv) {return !sv.empty(); }) | views::adjacent<2>) {
            l = string_view(l.data(), l.size() + r.size());
            r = {};
            break;
        }
    }
*/





    try {
        Ort::Env env { ORT_LOGGING_LEVEL_WARNING, "" };

        Scheduler::Create("I:/huggingface/hub/models--nmkd--stable-diffusion-1.5-onnx-fp16/snapshots/38dacf2c14c89e3538b5e32da888eb9c46e0e1bf/scheduler/scheduler_config.json");

        const auto start_time = chrono::high_resolution_clock::now();

        //const auto pipeline = Pipeline::Load(env, "I:/huggingface/hub/models--onnxruntime--sdxl-turbo/snapshots/bd6180e5aa5a5e326fbb0ba1bdda15cb3817f63c/");
        const auto pipeline = Pipeline::Load(env, "I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/");
        //const auto pipeline = Pipeline::Load(env, "I:/huggingface/hub/models--tlwu--stable-diffusion-xl-base-1.0-onnxruntime/snapshots/621ce78dc071ee3b4407467df9800ba6b7673224/");

        println("Models loaded in {}", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time));

        // EulerDiscreteScheduler
        // EulerAncestralScheduler
        // PNDMScheduler
        //PNDMScheduler scheduler({ .timestep_spacing = "trailing", .beta_schedule = "scaled_linear",
        //                .num_train_timesteps = 1000, .beta_start = 0.00085f, .beta_end = 0.012f, .skip_prk_steps = true });
        //pipeline->SetScheduler(scheduler);

        pipeline->Run("Fantasy artwork of an Asian female punk rock guitarist with short blonde curly hair, stocky build, small round glasses",
                      "",
                      50, 7.f, 3);// , {}, "input_512.png", .5f);
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

#endif
