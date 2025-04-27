#include <print>
#include <ranges>
#include <span>

#include "pndmscheduler.h"

using namespace std;


static std::vector<float> linspace(float start, float end, size_t steps) {
    std::vector<float> res(steps);

    for (size_t i = 0; i < steps; i++)
        res[i] = (start + (steps - (steps - i)) * ((end - start) / (steps - 1)));

    return res;
}


static std::vector<float> cumprod(std::vector<float>& input) {
    std::vector<float> res = input;

    for (size_t i = 1; i < res.size(); i++)
        res[i] = res[i] * res[i - 1];

    return res;
}


template<typename DataType>
PNDMScheduler<DataType>::PNDMScheduler(size_t num_train_timesteps,
                             float beta_start,
                             float beta_end,
                             std::string beta_schedule,
                             std::optional< std::vector<float> > trained_betas,
                             bool skip_prk_steps,
                             bool set_alpha_to_one,
                             std::string prediction_type,
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
        _betas = linspace(std::sqrt(beta_start), std::sqrt(beta_end), num_train_timesteps);
        for (auto& b : _betas)
            b *= b;
    }
    else {
        throw std::invalid_argument(beta_schedule + "is not implemented for PNDMScheduler");
    }

    _alphas = _betas;
    for (auto& a : _alphas)
        a = 1.f - a;

    _alphas_cumprod = cumprod(_alphas);

    if (set_alpha_to_one)
        _final_alpha_cumprod = 1.0f;
    else
        _final_alpha_cumprod = _alphas_cumprod[0];

    // standard deviation of the initial noise distribution
    _init_noise_sigma = 1.0f;

    //For now we only support F-PNDM, i.e. the runge-kutta method
    //For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
    //mainly at formula (9), (12), (13) and the Algorithm 2.
    _pndm_order = 4;

    __timesteps = views::iota(0LL, static_cast<int64_t>(num_train_timesteps)) | views::reverse | ranges::to<vector>();
}


template<typename DataType>
void PNDMScheduler<DataType>::set_timesteps(size_t num_inference_steps) {
    _num_inference_steps = num_inference_steps;

    int step_ratio = static_cast<int>(_num_train_timesteps / num_inference_steps);

    // creates integer timesteps by multiplying by ratio
    // casting to int to avoid issues when num_inference_step is power of 3
    //self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
   // self._timesteps += self.config.steps_offset
    __timesteps.resize(num_inference_steps);
    for (size_t i = 0; i < num_inference_steps; i++)
        __timesteps[i] = i * step_ratio + steps_offset;

    if (_skip_prk_steps) {
        const auto reverted = views::reverse(__timesteps);
        _plms_timesteps.reserve(__timesteps.size() + 1);
        _plms_timesteps.assign({ reverted[0], reverted[1] });
        _plms_timesteps.append_range(views::drop(reverted, 1));
    }
    else {
        throw std::invalid_argument("set_timesteps is not yet implemented for skip_prk_steps=false config");
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
        throw std::invalid_argument("Not yet implemented step_prk method. Set skip_prk_steps to true for now.");
    }
    else {
        step_plms(model_output, timestep, sample);
    }
}


template<typename DataType>
void PNDMScheduler<DataType>::step_plms(span<DataType> model_output, int64_t timestep, span<DataType> sample) {
    if (!_num_inference_steps) {
        throw std::invalid_argument("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler");
    }

    if (!_skip_prk_steps && _ets.size() < 3) {
        throw std::invalid_argument("an only be run AFTER scheduler has been run ");
    }

    int64_t prev_timestep = timestep - _num_train_timesteps / *_num_inference_steps;

    if (_counter != 1) {
        while (_ets.size() > 3) _ets.pop_front();

        // when caching model_output, we need to make a deep copy to avoid the case where
        // caller retains shallow copy, and it modified by them.
        _ets.emplace_back(model_output.begin(), model_output.end());
    }
    else {
        prev_timestep = timestep;
        timestep = timestep + _num_train_timesteps / *_num_inference_steps;
    }

    if ((_ets.size() == 1) && _counter == 0) {
        _cur_sample.assign_range(sample);
    }
    else if ((_ets.size() == 1) && _counter == 1) {
        const auto pEts = _ets.back().data();

        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = DataType((static_cast<float>(model_output[i]) + pEts[i]) / 2.f);

        ranges::copy(_cur_sample, sample.data());
    }
    else if (_ets.size() == 2) {
        //model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        const auto pEtsM1 = _ets[1].data();
        const auto pEtsM2 = _ets[0].data();
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = DataType((3.f * pEtsM1[i] - pEtsM2[i]) / 2.f);
    }
    else if (_ets.size() == 3) {
        //model_output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        const auto pEtsM1 = _ets[2].data();
        const auto pEtsM2 = _ets[1].data();
        const auto pEtsM3 = _ets[0].data();
        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = DataType((23.f * pEtsM1[i] - 16.f * pEtsM2[i] + 5.f * pEtsM3[i]) / 12.f);
    }
    else {
        const auto pEtsM1 = (_ets.end() - 1)->data();
        const auto pEtsM2 = (_ets.end() - 2)->data();
        const auto pEtsM3 = (_ets.end() - 3)->data();
        const auto pEtsM4 = (_ets.end() - 4)->data();

        for (size_t i = 0; i < model_output.size(); i++)
            model_output[i] = DataType((1.f / 24.f) * (55.f * pEtsM1[i] - 59.f * pEtsM2[i] + 37.f * pEtsM3[i] - 9.f * pEtsM4[i]));
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
        throw std::invalid_argument("v_prediction case needs to be implemented.");
    }
    else if (_prediction_type != "epsilon") {
        throw std::invalid_argument("prediction_type given as " + _prediction_type + " must be one of `epsilon` or `v_prediction`");
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
    const auto sqrt_alpha_prod = std::sqrt(_alphas_cumprod[timestep]);
    const auto sqrt_one_minus_alpha_prod = std::sqrt(1 - _alphas_cumprod[timestep]);

    //be cautious here, keeping in mind that pNoise may be equal to pNoisySamples
    for (const auto& [sample, noise] : views::zip(samples, noise)) {
        const auto noisy_sample = sqrt_alpha_prod * static_cast<float>(sample) + sqrt_one_minus_alpha_prod * static_cast<float>(noise);
        sample = DataType(noisy_sample);
    }
}


template PNDMScheduler<Ort::Float16_t>;
template PNDMScheduler<float>;
