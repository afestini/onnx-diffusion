#include <print>
#include <ranges>
#include <span>

#include "pndmscheduler.h"

using namespace std;


static vector<float> linspace(float start, float end, size_t steps) {
    vector<float> res(steps);

    for (size_t i = 0; i < steps; i++)
        res[i] = (start + (steps - (steps - i)) * ((end - start) / (steps - 1)));

    return res;
}


static vector<float> cumprod(vector<float>& input) {
    vector<float> res = input;

    for (size_t i = 1; i < res.size(); i++)
        res[i] = res[i] * res[i - 1];

    return res;
}


template<typename DataType>
PNDMScheduler<DataType>::PNDMScheduler(size_t num_train_timesteps,
                                       float beta_start,
                                       float beta_end,
                                       string beta_schedule,
                                       optional< vector<float> > trained_betas,
                                       bool skip_prk_steps,
                                       bool set_alpha_to_one,
                                       string prediction_type,
                                       size_t steps_offset)
    : _num_train_timesteps(num_train_timesteps), steps_offset(steps_offset), _skip_prk_steps(skip_prk_steps), _prediction_type(prediction_type)
{
    if (trained_betas) {
        _betas = *trained_betas;
    }
    else if (beta_schedule == "linear") {
        _betas = linspace(beta_start, beta_end, num_train_timesteps);
    }
    else if (beta_schedule == "scaled_linear") {
        _betas = linspace(sqrt(beta_start), sqrt(beta_end), num_train_timesteps);
        for (auto& b : _betas)
            b *= b;
    }
    else {
        throw invalid_argument(beta_schedule + "is not implemented for PNDMScheduler");
    }

    _alphas = _betas;
    for (auto& a : _alphas)
        a = 1.f - a;

    _alphas_cumprod = cumprod(_alphas);

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

    const int step_ratio = static_cast<int>(_num_train_timesteps / num_inference_steps);

    vector<int64_t> tmp_timesteps(num_inference_steps);
    for (size_t i = 0; i < num_inference_steps; i++)
        tmp_timesteps[i] = steps_offset + i * step_ratio;

    if (_skip_prk_steps) {
        const auto reversed = views::reverse(tmp_timesteps);
        _plms_timesteps.reserve(reversed.size() + 1);
        _plms_timesteps.assign({ reversed[0], reversed[1] });
        _plms_timesteps.append_range(views::drop(reversed, 1));
    }
    else {
        throw invalid_argument("set_timesteps is not yet implemented for skip_prk_steps=false config");
    }

    _timesteps.reserve(_prk_timesteps.size() + _plms_timesteps.size());
    _timesteps.assign_range(_prk_timesteps);
    _timesteps.append_range(_plms_timesteps);

    _ets = {};
    _counter = 0;
}


template<typename DataType>
void PNDMScheduler<DataType>::step(span<DataType> model_output, int64_t timestep, span<DataType> sample) {
    if (_counter < ssize(_prk_timesteps) && !(_skip_prk_steps)) {
        throw invalid_argument("Not yet implemented step_prk method. Set skip_prk_steps to true for now.");
    }
    else {
        step_plms(model_output, timestep, sample);
    }
}


template<typename DataType>
void PNDMScheduler<DataType>::step_plms(span<DataType> model_output, int64_t timestep, span<DataType> sample) {
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
            model_output[i] = DataType((static_cast<float>(model_output[i]) + _ets[0][i]) / 2.f);

        ranges::copy(_cur_sample, sample.data());
        _cur_sample.clear();
    }
    else if (_ets.size() == 2) {
        //model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = DataType((3.f * _ets[1][i] - _ets[0][i]) / 2.f);
    }
    else if (_ets.size() == 3) {
        //model_output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = DataType((23.f * _ets[2][i] - 16.f * _ets[1][i] + 5.f * _ets[0][i]) / 12.f);
    }
    else {
        //model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = DataType((1.f / 24.f) * (55.f * _ets[3][i] - 59.f * _ets[2][i] + 37.f * _ets[1][i] - 9.f * _ets[0][i]));
    }

    _get_prev_sample(sample, timestep, prev_timestep, model_output);

    _counter += 1;
}


template<typename DataType>
void PNDMScheduler<DataType>::scale_model_input(span<DataType> sample, int64_t ts) { /* no-op in case of PNDM */ }


template<typename DataType>
void PNDMScheduler<DataType>::_get_prev_sample(span<DataType> sample, int64_t timestep, int64_t prev_timestep,
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
    for (const auto& [sample_val, model] : views::zip(sample, model_output)) {
        const auto new_sample = sample_coeff * static_cast<float>(sample_val) - (alpha_prod_t_prev - alpha_prod_t) * static_cast<float>(model) / model_output_denom_coeff;
        sample_val = DataType(new_sample);
    }
}


template<typename DataType>
void PNDMScheduler<DataType>::add_noise_to_sample(span<DataType> samples, span<const DataType> noise, int64_t timestep) {
    const auto sqrt_alpha_prod = sqrt(_alphas_cumprod[timestep]);
    const auto sqrt_one_minus_alpha_prod = sqrt(1 - _alphas_cumprod[timestep]);

    //be cautious here, keeping in mind that pNoise may be equal to pNoisySamples
    for (const auto& [sample, noise] : views::zip(samples, noise)) {
        const auto noisy_sample = sqrt_alpha_prod * static_cast<float>(sample) + sqrt_one_minus_alpha_prod * static_cast<float>(noise);
        sample = DataType(noisy_sample);
    }
}


template PNDMScheduler<Ort::Float16_t>;
template PNDMScheduler<float>;







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
        _betas = *trained_betas;
    }
    else if (beta_schedule == "linear") {
        _betas = linspace(beta_start, beta_end, num_train_timesteps);
    }
    else if (beta_schedule == "scaled_linear") {
        _betas = linspace(sqrt(beta_start), sqrt(beta_end), num_train_timesteps);
        for (auto& b : _betas)
            b *= b;
    }
    else {
        throw invalid_argument(beta_schedule + "is not implemented for USTMScheduler");
    }

    _alphas = _betas;
    for (auto& a : _alphas)
        a = 1.f - a;

    _alphas_cumprod = cumprod(_alphas);

    if (set_alpha_to_one)
        _final_alpha_cumprod = 1.0f;
    else
        _final_alpha_cumprod = _alphas_cumprod[0];
}


template<typename DataType>
void USTMScheduler<DataType>::set_timesteps(size_t num_inference_steps) {
    _num_inference_steps = num_inference_steps;

    const int step_ratio = static_cast<int>(_num_train_timesteps / num_inference_steps);

    _timesteps.resize(num_inference_steps);
    for (size_t i = 0; i < _timesteps.size(); i++)
        _timesteps[_timesteps.size() - i - 1] = i * step_ratio + steps_offset;
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
void USTMScheduler<DataType>::_get_prev_sample(span<DataType> sample,
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
    for (const auto& [sample_val, model] : views::zip(sample, model_output)) {
        const auto new_sample = sample_coeff * static_cast<float>(sample_val) - (alpha_prod_t_prev - alpha_prod_t) * static_cast<float>(model) / model_output_denom_coeff;
        sample_val = DataType(new_sample);
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
