#pragma once

#include <deque>
#include <span>
#include <string>

#include <onnxruntime_cxx_api.h>


template<typename DataType>
class PNDMScheduler {
public:
    PNDMScheduler(int64_t num_train_timesteps = 1000,
                  float beta_start = 0.0001f,
                  float beta_end = 0.002f,
                  std::string beta_schedule = "linear",
                  std::optional<std::vector<float>> trained_betas = {},
                  bool skip_prk_steps = false,
                  bool set_alpha_to_one = false,
                  std::string prediction_type = "epsilon",
                  int64_t steps_offset = 0);

    void reset();

    void set_timesteps(size_t num_inference_steps);

    int64_t get_steps_offset() { return _steps_offset; }

    void step(std::span<float> model_output, int64_t timestep, std::span<float> sample);

    void add_noise_to_sample(std::span<float> samples, std::span<const float> noise, int64_t timesteps);

    void scale_model_input(std::span<const float> sample, std::span<DataType> model_input, int64_t ts);

    float init_noise_sigma() const { return _init_noise_sigma; }

    const std::vector<int64_t>& timesteps() { return _timesteps; }

private:
    void step_plms(std::span<float> model_output, int64_t timestep, std::span<float> sample);

    void _get_prev_sample(std::span<float> sample, int64_t timestep, int64_t prev_timestep,
                          std::span<const float> model_output);

    std::vector<float> _alphas_cumprod;
    float _final_alpha_cumprod = 0.f;

    std::vector<int64_t> _timesteps;

    std::deque<std::vector<float>> _ets;
    std::vector<float> _cur_sample;

    int64_t _counter = 0;

    float _init_noise_sigma = 1.0;
    int _pndm_order = 4;

    int64_t _num_inference_steps = 0;

    int64_t _num_train_timesteps = 1000;
    int64_t _steps_offset = 0;

    bool _skip_prk_steps = false;

    std::string _prediction_type;
};





template<typename DataType>
class EulerDiscreteScheduler {
public:
    EulerDiscreteScheduler(size_t num_train_timesteps = 1000,
                           float beta_start = 0.0001f,
                           float beta_end = 0.002f,
                           std::string beta_schedule = "linear",
                           std::optional< std::vector<float> > trained_betas = {},
                           bool skip_prk_steps = false,
                           bool set_alpha_to_one = false,
                           std::string prediction_type = "epsilon",
                           size_t steps_offset = 0);

    void reset() {}

    void set_timesteps(size_t num_inference_steps);

    size_t get_steps_offset() { return _steps_offset; }

    void step(std::span<float> model_output, int64_t timestep, std::span<float> sample);

    void add_noise_to_sample(std::span<float> samples, std::span<const float> noise, int64_t timesteps);

    void scale_model_input(std::span<const float> in_sample, std::span<DataType> out_sample, int64_t ts);

    float init_noise_sigma() { return _init_noise_sigma; }

    const std::vector<int64_t>& timesteps() { return _timesteps; }

private:
    std::vector<float> _alphas_cumprod;
    std::vector<float> _sigmas;

    std::vector<int64_t> _timesteps;
    float _init_noise_sigma = 1.0;

    size_t _num_train_timesteps = 1000;
    size_t _num_inference_steps = 0;
    size_t _steps_offset = 0;
};





template<typename DataType>
class EulerAncestralScheduler {
public:
    EulerAncestralScheduler(size_t num_train_timesteps = 1000,
                            float beta_start = 0.0001,
                            float beta_end = 0.002,
                            std::string beta_schedule = "linear",
                            std::optional< std::vector<float> > trained_betas = {},
                            bool skip_prk_steps = false,
                            bool set_alpha_to_one = false,
                            std::string prediction_type = "epsilon",
                            size_t steps_offset = 0);

    void reset() {}

    void set_timesteps(size_t num_inference_steps);

    size_t get_steps_offset() { return 0; }

    void step(std::span<float> model_output, int64_t timestep, std::span<float> sample);

    void add_noise_to_sample(std::span<float> samples, std::span<const float> noise, int64_t timesteps);

    void scale_model_input(std::span<const float> sample, std::span<DataType> model_input, int64_t ts);

    float init_noise_sigma() { return _init_noise_sigma; }

    const std::vector<int64_t>& timesteps() { return _timesteps; }

private:
    std::vector<float> _sigmas;

    std::vector<int64_t> _timesteps;
    float _init_noise_sigma = 1.0;

    size_t _num_inference_steps = 0;
    size_t _num_train_timesteps = 1000;
    size_t _steps_offset = 0;

    std::string _prediction_type;
};
