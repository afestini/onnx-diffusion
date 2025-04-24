#pragma once

#include <deque>

#include <onnxruntime_cxx_api.h>


class PNDMScheduler {
public:
    PNDMScheduler(size_t num_train_timesteps = 1000,
        float beta_start = 0.0001f,
        float beta_end = 0.002f,
        std::string beta_schedule = "linear",
        std::optional<std::vector<float>> trained_betas = {},
        bool skip_prk_steps = false,
        bool set_alpha_to_one = false,
        std::string prediction_type = "epsilon",
        size_t steps_offset = 0);

    void set_timesteps(size_t num_inference_steps);

    size_t get_steps_offset() { return steps_offset; };

    void step(std::span<Ort::Float16_t> model_output, int64_t timestep, std::span<Ort::Float16_t> sample);

    void add_noise_to_sample(std::span<Ort::Float16_t> samples, std::span<const Ort::Float16_t> noise, int64_t timesteps);

    void scale_model_input(std::span<Ort::Float16_t> sample, int64_t ts);

    float init_noise_sigma() const { return _init_noise_sigma; }

    std::vector<int64_t> timesteps() { return _timesteps; }

private:
    void step_plms(std::span<Ort::Float16_t> model_output, int64_t timestep, std::span<Ort::Float16_t> sample);

    void _get_prev_sample(std::span<Ort::Float16_t> sample, int64_t timestep, int64_t prev_timestep,
                          std::span<const Ort::Float16_t> model_output);

    std::vector<float> _betas;
    std::vector<float> _alphas;
    std::vector<float> _alphas_cumprod;
    float _final_alpha_cumprod = 0.f;

    std::vector<int64_t> __timesteps;
    std::vector<int64_t> _timesteps;

    std::vector<int64_t> _prk_timesteps;
    std::vector<int64_t> _plms_timesteps;

    std::deque<std::vector<float>> _ets;
    std::vector<Ort::Float16_t> _cur_sample;

    int64_t _counter = 0;

    float _init_noise_sigma = 1.0;
    int _pndm_order = 4;

    std::optional<size_t> _num_inference_steps = {};

    size_t _num_train_timesteps;
    size_t steps_offset;

    bool _skip_prk_steps;

    std::string _prediction_type;
};
