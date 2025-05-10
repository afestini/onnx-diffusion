#include <algorithm>
#include <print>
#include <random>
#include <ranges>
#include <span>

#include "scheduler.h"
#include "json.h"

using namespace std;


template<typename T>
static vector<T> linspace(float start, float end, size_t steps) {
    vector<T> res(steps);

    if (steps == 1) return { static_cast<T>(end) };

    const auto step_size = (end - start) / (steps - 1);
    for (size_t i = 0; i + 1< steps; i++) {
        res[i] = T(start + i * step_size);
    }
    res.back() = T(end);
    return res;
}


static void make_cumprod(span<float> input) {
    for (size_t i = 1; i < input.size(); i++)
        input[i] *= input[i - 1];
}


static int64_t get_prev_timestep(const vector<int64_t> timesteps, int64_t timestep) {
    const auto it = ranges::upper_bound(timesteps, timestep, ranges::greater());
    return it != timesteps.end() ? *it : -1;
}


PNDMScheduler::PNDMScheduler(const SchedulerConfig& cfg)
    : Scheduler(cfg), _skip_prk_steps(cfg.skip_prk_steps), _prediction_type(cfg.prediction_type)
{
    if (cfg.beta_schedule == "linear") {
        _alphas_cumprod = linspace<float>(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps);
    }
    else if (cfg.beta_schedule == "scaled_linear") {
        _alphas_cumprod = linspace<float>(sqrt(cfg.beta_start), sqrt(cfg.beta_end), cfg.num_train_timesteps);
        for (auto& b : _alphas_cumprod)
            b *= b;
    }
    else {
        throw invalid_argument(cfg.beta_schedule + "is not implemented for PNDMScheduler");
    }

    for (auto& a : _alphas_cumprod)
        a = 1.f - a;

    make_cumprod(_alphas_cumprod);

    _final_alpha_cumprod = cfg.set_alpha_to_one ? 1.0f : _alphas_cumprod[0];
}


void PNDMScheduler::reset() {
    _ets.clear();
    _counter = 0;
}


void PNDMScheduler::set_timesteps(size_t num_inference_steps) {
    _num_inference_steps = num_inference_steps;

    vector<int64_t> tmp_timesteps;
    tmp_timesteps.reserve(num_inference_steps);

    const float step_ratio = static_cast<float>(_num_train_timesteps) / static_cast<float>(num_inference_steps);
    float value = static_cast<float>(_num_train_timesteps) - 1.f;
    while (value > 0.f) {
        tmp_timesteps.emplace_back(static_cast<int64_t>(round(value)));
        value -= step_ratio;
    }

    if (_skip_prk_steps) {
        if (num_inference_steps > 1) {
            _timesteps.reserve(tmp_timesteps.size() + 1);
            _timesteps.assign({ tmp_timesteps[0], tmp_timesteps[1] });
            _timesteps.append_range(views::drop(tmp_timesteps, 1));
        }
        else
            _timesteps = move(tmp_timesteps);
    }
    else {
        throw invalid_argument("set_timesteps is not yet implemented for skip_prk_steps=false config");
    }

    _ets = {};
    _counter = 0;
}


void PNDMScheduler::step(span<float> model_output, int64_t timestep, span<float> sample) {
    if (!_skip_prk_steps) {
        throw invalid_argument("Not yet implemented step_prk method. Set skip_prk_steps to true for now.");
    }
    else {
        step_plms(model_output, timestep, sample);
    }
}


void PNDMScheduler::step_plms(span<float> model_output, int64_t timestep, span<float> sample) {
    if (_num_inference_steps == 0) {
        throw invalid_argument("Number of inference steps is 0, you need to run 'set_timesteps' after creating the scheduler");
    }

    if (!_skip_prk_steps && _ets.size() < 3) {
        throw invalid_argument("an only be run AFTER scheduler has been run ");
    }

    int64_t prev_timestep = get_prev_timestep(_timesteps, timestep);

    if (_counter != 1) {
        while (_ets.size() > 3) _ets.pop_front();

        // when caching model_output, we need to make a deep copy to avoid the case where
        // caller retains shallow copy, and it modified by them.
        _ets.emplace_back(model_output.begin(), model_output.end());
    }
    else {
        prev_timestep = timestep;
        timestep = timestep + _num_train_timesteps / _num_inference_steps;
    }

    if ((_ets.size() == 1) && _counter == 0) {
        _cur_sample.assign_range(sample);
    }
    else if ((_ets.size() == 1) && _counter == 1) {
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = (model_output[i] + _ets[0][i]) / 2.f;

        ranges::copy(_cur_sample, sample.data());
        _cur_sample.clear();
    }
    else if (_ets.size() == 2) {
        //model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = (3.f * _ets[1][i] - _ets[0][i]) / 2.f;
    }
    else if (_ets.size() == 3) {
        //model_output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = (23.f * _ets[2][i] - 16.f * _ets[1][i] + 5.f * _ets[0][i]) / 12.f;
    }
    else {
        //model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = (1.f / 24.f) * (55.f * _ets[3][i] - 59.f * _ets[2][i] + 37.f * _ets[1][i] - 9.f * _ets[0][i]);
    }

    _get_prev_sample(sample, timestep, prev_timestep, model_output);

    _counter += 1;
}


void PNDMScheduler::_get_prev_sample(span<float> sample, int64_t timestep, int64_t prev_timestep, span<const float> model_output) const {
    // See formula(9) of PNDM paper https ://arxiv.org/pdf/2202.09778.pdf
    const auto alpha_prod_t = _alphas_cumprod[timestep];
    const float alpha_prod_t_prev = (prev_timestep >= 0) ? _alphas_cumprod[prev_timestep] : _final_alpha_cumprod;

    const auto beta_prod_t = 1.f - alpha_prod_t;
    const auto beta_prod_t_prev = 1.f - alpha_prod_t_prev;

    if (_prediction_type == "v_prediction") {
        throw invalid_argument("v_prediction case needs to be implemented.");
    }
    else if (_prediction_type != "epsilon") {
        throw invalid_argument("prediction_type given as " + _prediction_type + " must be one of `epsilon` or `v_prediction`");
    }

    const auto sample_coeff = sqrt(alpha_prod_t_prev / alpha_prod_t);

    const auto model_output_denom_coeff = alpha_prod_t * sqrt(beta_prod_t_prev) +
                                          sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);

    for (const auto& [sample_val, model] : views::zip(sample, model_output))
        sample_val = sample_coeff * sample_val - (alpha_prod_t_prev - alpha_prod_t) * model / model_output_denom_coeff;
}


void PNDMScheduler::add_noise_to_sample(span<float> samples, int64_t timestep) {
    const auto sqrt_alpha_prod = sqrt(_alphas_cumprod[timestep]);
    const auto sqrt_one_minus_alpha_prod = sqrt(1 - _alphas_cumprod[timestep]);

    for (auto& sample : samples)
        sample = sqrt_alpha_prod * sample + sqrt_one_minus_alpha_prod * rng_dist(rng);
}








EulerDiscreteScheduler::EulerDiscreteScheduler(const SchedulerConfig& cfg)
    : Scheduler(cfg)
{
    vector<float> betas;

    if (cfg.beta_schedule == "linear") {
        betas = linspace<float>(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps);
    }
    else if (cfg.beta_schedule == "scaled_linear") {
        betas = linspace<float>(sqrt(cfg.beta_start), sqrt(cfg.beta_end), cfg.num_train_timesteps);
        for (auto& b : betas)
            b *= b;
    }

    _sigmas.reserve(cfg.num_train_timesteps);

    float alpha_prod = 1.0f;
    for (const auto beta : betas) {
        alpha_prod *= (1.f - beta);
        _sigmas.push_back(sqrt((1.0f - alpha_prod) / alpha_prod));
    }
}


void EulerDiscreteScheduler::set_timesteps(size_t num_inference_steps) {
    _num_inference_steps = num_inference_steps;

    _timesteps.clear();
    _timesteps.reserve(num_inference_steps);

    if (_timestep_spacing == "linspace") {
        _timesteps = linspace<int64_t>(0.f, static_cast<float>(_num_train_timesteps - 1), num_inference_steps) | views::reverse | ranges::to<vector>();
    }
    else if (_timestep_spacing == "leading") {
        const int64_t step_ratio = _num_train_timesteps / num_inference_steps;
        for (auto i : views::iota(0ULL, num_inference_steps) | views::reverse)
            _timesteps.emplace_back(_steps_offset + i * step_ratio);
    }
    else if (_timestep_spacing == "trailing") {
        const float step_ratio = static_cast<float>(_num_train_timesteps) / static_cast<float>(num_inference_steps);
        float value = static_cast<float>(_num_train_timesteps) - 1.f;
        while (value > 0.f) {
            _timesteps.emplace_back(static_cast<int64_t>(round(value)));
            value -= step_ratio;
        }
    }

    _init_noise_sigma = _sigmas[_timesteps[0]];
    //_init_noise_sigma = sqrt(_init_noise_sigma * _init_noise_sigma + 1);
}


void EulerDiscreteScheduler::step(span<float> model_output, int64_t timestep, span<float> samples) {
    const int64_t prev_timestep = get_prev_timestep(_timesteps, timestep);

    const auto sigma = _sigmas[timestep];
    const auto sigma_prev = (prev_timestep >= 0) ? _sigmas[prev_timestep] : 0.f;

    for (const auto& [sample, model] : views::zip(samples, model_output)) {
        const auto pred_original_sample = sample - sigma * model;
        const auto derivative = (sample - pred_original_sample) / sigma;
        const auto dt = sigma_prev - sigma;

        sample = sample + derivative * dt;
    }
}


void EulerDiscreteScheduler::add_noise_to_sample(span<float> samples, int64_t timestep) {
    const auto sigma = _sigmas[timestep];
    for (auto& sample : samples)
        sample += rng_dist(rng) * sigma;
}


float EulerDiscreteScheduler::scale_model_input_factor(int64_t ts) const {
    const float sigma = _sigmas[ts];
    return 1.f / sqrt(sigma * sigma + 1);
}




EulerAncestralScheduler::EulerAncestralScheduler(const SchedulerConfig& cfg)
    : Scheduler(cfg), _prediction_type(cfg.prediction_type)
{
    vector<float> betas;

    if (cfg.beta_schedule == "linear") {
        betas = linspace<float>(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps);
    }
    else if (cfg.beta_schedule == "scaled_linear") {
        betas = linspace<float>(sqrt(cfg.beta_start), sqrt(cfg.beta_end), cfg.num_train_timesteps);
        for (auto& b : betas)
            b *= b;
    }

    _sigmas.reserve(cfg.num_train_timesteps + 1);
    _sigmas.push_back(0.f);

    float alpha_prod = 1.0f;
    for (const auto beta : betas) {
        alpha_prod *= (1.0f - beta);
        _sigmas.push_back(sqrt((1.0f - alpha_prod) / alpha_prod));
    }
}


void EulerAncestralScheduler::set_timesteps(size_t num_inference_steps) {
    _num_inference_steps = num_inference_steps;

    _timesteps.clear();
    _timesteps.reserve(num_inference_steps);

    if (_timestep_spacing == "linspace") {
        _timesteps = linspace<int64_t>(0.f, static_cast<float>(_num_train_timesteps - 1), num_inference_steps) | views::reverse | ranges::to<vector>();
    }
    else if (_timestep_spacing == "leading") {
        const int64_t step_ratio = _num_train_timesteps / num_inference_steps;
        for (auto i : views::iota(0ULL, num_inference_steps) | views::reverse)
            _timesteps.emplace_back(_steps_offset + i * step_ratio);
    }
    else if (_timestep_spacing == "trailing") {
        const float step_ratio = static_cast<float>(_num_train_timesteps) / static_cast<float>(num_inference_steps);
        float value = static_cast<float>(_num_train_timesteps) - 1.f;
        while (value > 0.f) {
            _timesteps.emplace_back(static_cast<int64_t>(round(value)));
            value -= step_ratio;
        }
    }

    _init_noise_sigma = _sigmas[_timesteps[0]];
    //_init_noise_sigma = sqrt(_init_noise_sigma * _init_noise_sigma + 1);
}


void EulerAncestralScheduler::step(span<float> model_output, int64_t timestep, span<float> samples) {
    const int64_t prev_timestep = get_prev_timestep(_timesteps, timestep);

    // +1 to compensate for the inserted 0 at the start
    const auto sigma = _sigmas[timestep + 1];
    const auto sigma_to = _sigmas[prev_timestep + 1];

    const auto sigma_from_sq = sigma * sigma;
    const auto sigma_to_sq = sigma_to * sigma_to;

    const auto sigma_up = sqrt(sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq);
    const auto sigma_down = sqrt(sigma_to_sq - (sigma_up * sigma_up));
    const auto dt = sigma_down - sigma;

    for (const auto& [sample, model] : views::zip(samples, model_output)) {
        const auto pred_org_sample = sample - sigma * model;
        const auto derivative = (sample - pred_org_sample) / sigma;
        sample += dt * derivative + sigma_up * rng_dist(rng);
    }
}


void EulerAncestralScheduler::add_noise_to_sample(span<float> samples, int64_t timestep) {
    const auto sigma = _sigmas[timestep];
    for (auto& sample : samples)
        sample += rng_dist(rng) * sigma;
}


float EulerAncestralScheduler::scale_model_input_factor(int64_t ts) const {
    const float sigma = _sigmas[ts + 1];
    return 1.f / sqrt(sigma * sigma + 1);
}



SchedulerConfig LoadSchedulerConfig(const std::filesystem::path& config_file) {
    SchedulerConfig cfg;
    JsonParser parser;
    const auto json = parser.Parse(config_file);
    cfg.class_name = json["_class_name"].as<string>();
    cfg.beta_end = json["beta_end"];
    cfg.beta_start = json["beta_start"];
    cfg.beta_schedule = json["beta_schedule"].as<string>();
    cfg.num_train_timesteps = json["num_train_timesteps"];
    cfg.prediction_type = json["prediction_type"].as<string>();
    cfg.set_alpha_to_one = json["set_alpha_to_one"];
    cfg.skip_prk_steps = json["skip_prk_steps"];
    cfg.steps_offset = json["steps_offset"];
    cfg.timestep_spacing = json["timestep_spacing"].as<string>();
    return cfg;
}


std::shared_ptr<Scheduler> Scheduler::Create(const filesystem::path& config_file) {
    const auto cfg = LoadSchedulerConfig(config_file);

    if (cfg.class_name == "PNDMScheduler")
        return make_shared<PNDMScheduler>(cfg);
    else if (cfg.class_name == "EulerDiscreteScheduler")
        return make_shared<EulerDiscreteScheduler>(cfg);
    else if (cfg.class_name == "EulerAncestralDiscreteScheduler")
        return make_shared<EulerAncestralScheduler>(cfg);

    return {};
}
