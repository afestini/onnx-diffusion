#include <chrono>
#include <print>

#include <onnxruntime_cxx_api.h>

#include "pipeline.h"
#include "scheduler.h"


using namespace std;


int main(int argc, char* argv[]) {
    try {
        Ort::Env env { ORT_LOGGING_LEVEL_WARNING, "" };

        Scheduler::Create("I:/huggingface/hub/models--nmkd--stable-diffusion-1.5-onnx-fp16/snapshots/38dacf2c14c89e3538b5e32da888eb9c46e0e1bf/scheduler/scheduler_config.json");

        const auto start_time = chrono::high_resolution_clock::now();

        //const auto pipeline = Pipeline::Load(env, "I:/huggingface/hub/models--onnxruntime--sdxl-turbo/snapshots/bd6180e5aa5a5e326fbb0ba1bdda15cb3817f63c/");
        //const auto pipeline = Pipeline::Load(env, "I:/huggingface/hub/models--sharpbai--stable-diffusion-v1-5-onnx-cuda-fp16/snapshots/e2ca53a1d64f7d181660cf6670c507c04cd5d265/");
        const auto pipeline = Pipeline::Load(env, "I:/huggingface/hub/models--tlwu--stable-diffusion-xl-base-1.0-onnxruntime/snapshots/621ce78dc071ee3b4407467df9800ba6b7673224/");

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
