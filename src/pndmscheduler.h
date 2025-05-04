#pragma once

#include <deque>
#include <random>
#include <span>
#include <string>

#include <onnxruntime_cxx_api.h>


class Scheduler {
public:
    Scheduler(int64_t num_train_timesteps, std::string timestep_spacing, int64_t steps_offset)
        : _num_train_timesteps(num_train_timesteps), _timestep_spacing(timestep_spacing), _steps_offset(steps_offset) {}

    virtual void set_timesteps(size_t num_inference_steps) = 0;

    virtual void step(std::span<float> model_output, int64_t timestep, std::span<float> sample) = 0;

    virtual void add_noise_to_sample(std::span<float> samples, int64_t timesteps) = 0;

    virtual float scale_model_input_factor(int64_t ts) const { return 1.f; }

    virtual void reset() {}

    int64_t get_steps_offset() const { return _steps_offset; }

    float init_noise_sigma() const { return _init_noise_sigma; }

    const std::vector<int64_t>& timesteps() const { return _timesteps; }

protected:
    std::mt19937 rng { std::random_device()() };
    std::normal_distribution<float> rng_dist { 0.f, 1.f };

    std::string _timestep_spacing;
    std::vector<int64_t> _timesteps;
    int64_t _num_train_timesteps = 1000;
    int64_t _num_inference_steps = 0;
    int64_t _steps_offset = 0;
    float _init_noise_sigma = 1.0;
};


class PNDMScheduler : public Scheduler {
public:
    PNDMScheduler(int64_t num_train_timesteps = 1000,
                  float beta_start = 0.0001f,
                  float beta_end = 0.002f,
                  std::string beta_schedule = "linear",
                  std::string timestep_spacing = "trailing",
                  bool skip_prk_steps = false,
                  bool set_alpha_to_one = false,
                  std::string prediction_type = "epsilon",
                  int64_t steps_offset = 0);

    void reset() override;

    void set_timesteps(size_t num_inference_steps) override;

    void step(std::span<float> model_output, int64_t timestep, std::span<float> sample) override;

    void add_noise_to_sample(std::span<float> samples, int64_t timesteps) override;

private:
    void step_plms(std::span<float> model_output, int64_t timestep, std::span<float> sample);

    void _get_prev_sample(std::span<float> sample, int64_t timestep, int64_t prev_timestep,
                          std::span<const float> model_output) const;

    std::vector<float> _alphas_cumprod;
    float _final_alpha_cumprod = 0.f;
    std::deque<std::vector<float>> _ets;

    std::vector<float> _cur_sample;
    int64_t _counter = 0;
    int _pndm_order = 4;
     bool _skip_prk_steps = false;
    std::string _prediction_type;
};


class EulerDiscreteScheduler : public Scheduler {
public:
    EulerDiscreteScheduler(size_t num_train_timesteps = 1000,
                           float beta_start = 0.0001f,
                           float beta_end = 0.002f,
                           std::string beta_schedule = "linear",
                           std::string timestep_spacing = "leading",
                           bool skip_prk_steps = false,
                           bool set_alpha_to_one = false,
                           std::string prediction_type = "epsilon",
                           size_t steps_offset = 0);

    void set_timesteps(size_t num_inference_steps) override;

    void step(std::span<float> model_output, int64_t timestep, std::span<float> sample) override;

    void add_noise_to_sample(std::span<float> samples, int64_t timesteps) override;

    float scale_model_input_factor(int64_t ts) const override;

private:
    std::vector<float> _sigmas;
};


class EulerAncestralScheduler : public Scheduler {
public:
    EulerAncestralScheduler(size_t num_train_timesteps = 1000,
                            float beta_start = 0.0001,
                            float beta_end = 0.002,
                            std::string beta_schedule = "linear",
                            std::string timestep_spacing = "trailing",
                            bool skip_prk_steps = false,
                            bool set_alpha_to_one = false,
                            std::string prediction_type = "epsilon",
                            size_t steps_offset = 0);

    void set_timesteps(size_t num_inference_steps) override;

    void step(std::span<float> model_output, int64_t timestep, std::span<float> sample) override;

    void add_noise_to_sample(std::span<float> samples, int64_t timesteps) override;

    float scale_model_input_factor(int64_t ts) const override;

private:
    std::vector<float> _sigmas;
    std::string _prediction_type;
};
