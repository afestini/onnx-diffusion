#include <algorithm>
#include <print>
#include <random>
#include <ranges>
#include <span>

#include "pndmscheduler.h"

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


template<typename T>
static void make_cumprod(span<T> input) {
    for (size_t i = 1; i < input.size(); i++)
        input[i] *= input[i - 1];
}


template<typename DataType>
PNDMScheduler<DataType>::PNDMScheduler(int64_t num_train_timesteps,
                                       float beta_start,
                                       float beta_end,
                                       string beta_schedule,
                                       optional<vector<float>> trained_betas,
                                       bool skip_prk_steps,
                                       bool set_alpha_to_one,
                                       string prediction_type,
                                       int64_t steps_offset)
    : _num_train_timesteps(num_train_timesteps), _steps_offset(steps_offset), _skip_prk_steps(skip_prk_steps), _prediction_type(prediction_type)
{
    if (trained_betas) {
        _alphas_cumprod = move(*trained_betas);
    }
    else if (beta_schedule == "linear") {
        _alphas_cumprod = linspace<float>(beta_start, beta_end, num_train_timesteps);
    }
    else if (beta_schedule == "scaled_linear") {
        _alphas_cumprod = linspace<float>(sqrt(beta_start), sqrt(beta_end), num_train_timesteps);
        for (auto& b : _alphas_cumprod)
            b *= b;
    }
    else {
        throw invalid_argument(beta_schedule + "is not implemented for PNDMScheduler");
    }

    for (auto& a : _alphas_cumprod)
        a = 1.f - a;

    make_cumprod<float>(_alphas_cumprod);

    if (set_alpha_to_one)
        _final_alpha_cumprod = 1.0f;
    else
        _final_alpha_cumprod = _alphas_cumprod[0];
}


template<typename DataType>
void PNDMScheduler<DataType>::reset() {
    _ets.clear();
    _counter = 0;
}


template<typename DataType>
void PNDMScheduler<DataType>::set_timesteps(size_t num_inference_steps) {
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


template<typename DataType>
void PNDMScheduler<DataType>::step(span<float> model_output, int64_t timestep, span<float> sample) {
    if (!_skip_prk_steps) {
        throw invalid_argument("Not yet implemented step_prk method. Set skip_prk_steps to true for now.");
    }
    else {
        step_plms(model_output, timestep, sample);
    }
}


template<typename DataType>
void PNDMScheduler<DataType>::step_plms(span<float> model_output, int64_t timestep, span<float> sample) {
    if (_num_inference_steps == 0) {
        throw invalid_argument("Number of inference steps is 0, you need to run 'set_timesteps' after creating the scheduler");
    }

    if (!_skip_prk_steps && _ets.size() < 3) {
        throw invalid_argument("an only be run AFTER scheduler has been run ");
    }

    int64_t prev_timestep = timestep - _num_train_timesteps / _num_inference_steps;

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


template<typename DataType>
void PNDMScheduler<DataType>::scale_model_input(span<const float> in_sample, span<DataType> out_sample, int64_t ts) {
    for (const auto& [in, out] : views::zip(in_sample, out_sample))
        out = DataType(in);
}


template<typename DataType>
void PNDMScheduler<DataType>::_get_prev_sample(span<float> sample, int64_t timestep, int64_t prev_timestep, span<const float> model_output) {
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


template<typename DataType>
void PNDMScheduler<DataType>::add_noise_to_sample(span<float> samples, span<const float> noise, int64_t timestep) {
    const auto sqrt_alpha_prod = sqrt(_alphas_cumprod[timestep]);
    const auto sqrt_one_minus_alpha_prod = sqrt(1 - _alphas_cumprod[timestep]);

    //be cautious here, keeping in mind that pNoise may be equal to pNoisySamples
    for (const auto& [sample, noise] : views::zip(samples, noise))
        sample = sqrt_alpha_prod * sample + sqrt_one_minus_alpha_prod * noise;
}


template PNDMScheduler<Ort::Float16_t>;
template PNDMScheduler<float>;








template<typename DataType>
EulerDiscreteScheduler<DataType>::EulerDiscreteScheduler(size_t num_train_timesteps,
                                                         float beta_start,
                                                         float beta_end,
                                                         string beta_schedule,
                                                         optional< vector<float> > trained_betas,
                                                         bool skip_prk_steps,
                                                         bool set_alpha_to_one,
                                                         string,
                                                         size_t steps_offset)
    : _num_train_timesteps(num_train_timesteps), _steps_offset(steps_offset)
{
    vector<float> betas;

    if (trained_betas) {
        betas = move(*trained_betas);
    }
    else if (beta_schedule == "linear") {
        betas = linspace<float>(beta_start, beta_end, num_train_timesteps);
    }
    else if (beta_schedule == "scaled_linear") {
        betas = linspace<float>(sqrt(beta_start), sqrt(beta_end), num_train_timesteps);
        for (auto& b : betas)
            b *= b;
    }

    _alphas_cumprod.reserve(num_train_timesteps);
    _sigmas.reserve(num_train_timesteps);

    float alpha_prod = 1.0f;
    for (const auto beta : betas) {
        alpha_prod *= (1.f - beta);
        _alphas_cumprod.push_back(sqrt(alpha_prod));
        //_sigmas.push_back(sqrt(1.0f - alpha_prod));
        _sigmas.push_back(sqrt((1.0f - alpha_prod) / alpha_prod));
    }

    _init_noise_sigma = _sigmas.back(); // sqrt(_sigmas.back() * _sigmas.back() + 1);
}


template<typename DataType>
void EulerDiscreteScheduler<DataType>::set_timesteps(size_t num_inference_steps) {
    _num_inference_steps = num_inference_steps;

    const size_t step_ratio = _num_train_timesteps / num_inference_steps;

    _timesteps.resize(num_inference_steps);
    for (size_t i = 0; i < _timesteps.size(); i++)
        _timesteps[i] = _num_train_timesteps - 1 - i * step_ratio;
}


template<typename DataType>
void EulerDiscreteScheduler<DataType>::step(span<float> model_output, int64_t timestep, span<float> samples) {
    const int64_t prev_timestep = timestep - (_num_train_timesteps / _num_inference_steps);

    const auto sigma = _sigmas[timestep];
    const auto sigma_prev = (prev_timestep >= 0) ? _sigmas[prev_timestep] : 0.f;

    const float alpha_cumprod_t = _alphas_cumprod[timestep];
    const float sigma_t = _sigmas[timestep];

    for (const auto& [sample, model] : views::zip(samples, model_output)) {
        const auto pred_original_sample = sample - sigma * model;
        const auto derivative = (sample - pred_original_sample) / sigma;

        const auto dt = sigma_prev - sigma;
        sample = sample + derivative * dt;
    }
}


template<typename DataType>
void EulerDiscreteScheduler<DataType>::add_noise_to_sample(span<float> samples, span<const float> noise, int64_t timestep) {}


template<typename DataType>
void EulerDiscreteScheduler<DataType>::scale_model_input(span<const float> in_sample, span<DataType> out_sample, int64_t ts) {
    const float sigma = _sigmas[ts];
    const auto factor = 1.f / sqrt(sigma * sigma + 1);
    for (const auto& [in, out] : views::zip(in_sample, out_sample))
        out = DataType(in * factor);
}


template EulerDiscreteScheduler<Ort::Float16_t>;
template EulerDiscreteScheduler<float>;





template<typename DataType>
EulerAncestralScheduler<DataType>::EulerAncestralScheduler(size_t num_train_timesteps,
                                                           float beta_start,
                                                           float beta_end,
                                                           string beta_schedule,
                                                           optional<vector<float>> trained_betas,
                                                           bool skip_prk_steps,
                                                           bool set_alpha_to_one,
                                                           string prediction_type,
                                                           size_t steps_offset)
    : _num_train_timesteps(num_train_timesteps), _prediction_type(prediction_type), _steps_offset(steps_offset)
{
    vector<float> betas;

    if (trained_betas) {
        betas = move(*trained_betas);
    }
    else if (beta_schedule == "linear") {
        betas = linspace<float>(beta_start, beta_end, num_train_timesteps);
    }
    else if (beta_schedule == "scaled_linear") {
        betas = linspace<float>(sqrt(beta_start), sqrt(beta_end), num_train_timesteps);
        for (auto& b : betas)
            b *= b;
    }

    _sigmas.reserve(num_train_timesteps + 1);
    _sigmas.push_back(0.f);

    float alpha_prod = 1.0f;
    for (const auto beta : betas) {
        alpha_prod *= (1.0f - beta);
        _sigmas.push_back(sqrt((1.0f - alpha_prod) / alpha_prod));
    }

    _init_noise_sigma = _sigmas.back();
}


template<typename DataType>
void EulerAncestralScheduler<DataType>::set_timesteps(size_t num_inference_steps) {
    _num_inference_steps = num_inference_steps;

    _timesteps.clear();
    _timesteps.reserve(num_inference_steps);

    // Spacing "leading"
    if (false) {
        const int64_t step_ratio = _num_train_timesteps / num_inference_steps;
        for (auto i : views::iota(0ULL, num_inference_steps) | views::reverse)
            _timesteps.emplace_back(i * step_ratio + _steps_offset);
    }
    // Spacing "trailing"
    const float step_ratio = static_cast<float>(_num_train_timesteps) / static_cast<float>(num_inference_steps);
    float value = static_cast<float>(_num_train_timesteps) - 1.f;
    while (value > 0.f) {
        _timesteps.emplace_back(static_cast<int64_t>(round(value)));
        value -= step_ratio;
    }
}


template<typename DataType>
void EulerAncestralScheduler<DataType>::step(span<float> model_output, int64_t timestep, span<float> samples) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.f, 1.f);

    const int64_t prev_timestep = timestep - (_num_train_timesteps / _num_inference_steps);

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
        sample += dt * derivative + sigma_up * dist(gen);
    }
}


template<typename DataType>
void EulerAncestralScheduler<DataType>::add_noise_to_sample(span<float> samples, span<const float> noise, int64_t timestep) {}


template<typename DataType>
void EulerAncestralScheduler<DataType>::scale_model_input(span<const float> in_sample, span<DataType> out_sample, int64_t ts) {
    const float sigma = _sigmas[ts + 1];
    const auto factor = 1.f / sqrt(sigma * sigma + 1);
    for (const auto& [in, out] : views::zip(in_sample, out_sample))
        out = DataType(in * factor);
}


template EulerAncestralScheduler<Ort::Float16_t>;
template EulerAncestralScheduler<float>;











template<typename DataType>
USTMScheduler<DataType>::USTMScheduler(size_t num_train_timesteps,
                                       float beta_start,
                                       float beta_end,
                                       string beta_schedule,
                                       optional< vector<float> > trained_betas,
                                       bool set_alpha_to_one,
                                       string prediction_type,
                                       size_t steps_offset)
    : _num_train_timesteps(num_train_timesteps), steps_offset(steps_offset), _prediction_type(prediction_type)
{
    if (trained_betas) {
        _alphas_cumprod = move(*trained_betas);
    }
    else if (beta_schedule == "linear") {
        _alphas_cumprod = linspace<float>(beta_start, beta_end, num_train_timesteps);
    }
    else if (beta_schedule == "scaled_linear") {
        _alphas_cumprod = linspace<float>(sqrt(beta_start), sqrt(beta_end), num_train_timesteps);
        for (auto& b : _alphas_cumprod)
            b *= b;
    }
    else {
        throw invalid_argument(beta_schedule + "is not implemented for USTMScheduler");
    }

    for (auto& a : _alphas_cumprod)
        a = 1.f - a;

    make_cumprod<float>(_alphas_cumprod);

    if (set_alpha_to_one)
        _final_alpha_cumprod = 1.0f;
    else
        _final_alpha_cumprod = _alphas_cumprod[0];
}


template<typename DataType>
void USTMScheduler<DataType>::set_timesteps(size_t num_inference_steps) {
    _num_inference_steps = num_inference_steps;

    const size_t step_ratio = _num_train_timesteps / num_inference_steps;

    _timesteps.resize(num_inference_steps);
    for (size_t i = 0; i < _timesteps.size(); i++)
        _timesteps[i] = _num_train_timesteps - 1 - i * step_ratio;
}


template<typename DataType>
void USTMScheduler<DataType>::step(span<DataType> model_output, int64_t timestep, span<DataType> sample) {
    if (_num_inference_steps == 0) {
        throw invalid_argument("Number of inference steps is 0, you need to run 'set_timesteps' after creating the scheduler");
    }

    int64_t prev_timestep = timestep - _num_train_timesteps / _num_inference_steps;
    _get_prev_sample(sample, timestep, prev_timestep, model_output);
}


template<typename DataType>
void USTMScheduler<DataType>::_get_prev_sample(span<DataType> samples,
                                               int64_t timestep,
                                               int64_t prev_timestep,
                                               span<const DataType> model_output) {
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

    // full formula(9)
    //    prev_sample = sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
    for (const auto& [sample, model] : views::zip(samples, model_output)) {
        const auto new_sample = sample_coeff * static_cast<float>(sample) - (alpha_prod_t_prev - alpha_prod_t) * static_cast<float>(model) / model_output_denom_coeff;
        sample = DataType(new_sample);
    }
}


template<typename DataType>
void USTMScheduler<DataType>::add_noise_to_sample(span<DataType> samples, span<const DataType> noise, int64_t timestep) {
    const auto sqrt_alpha_prod = sqrt(_alphas_cumprod[timestep]);
    const auto sqrt_one_minus_alpha_prod = sqrt(1 - _alphas_cumprod[timestep]);

    //be cautious here, keeping in mind that pNoise may be equal to pNoisySamples
    for (const auto& [sample, noise] : views::zip(samples, noise)) {
        const auto noisy_sample = sqrt_alpha_prod * static_cast<float>(sample) + sqrt_one_minus_alpha_prod * static_cast<float>(noise);
        sample = DataType(noisy_sample);
    }
}


template<typename DataType>
void USTMScheduler<DataType>::scale_model_input(span<DataType> sample, int64_t ts) {}


template USTMScheduler<Ort::Float16_t>;
template USTMScheduler<float>;
