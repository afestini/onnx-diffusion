#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

class Scheduler;

namespace Ort {
struct Env;
}


class Pipeline {
public:
    static std::shared_ptr<Pipeline> Load(const Ort::Env& env, const std::string& root);

    Pipeline(const Ort::Env& environment) : env(&environment) {}
    virtual ~Pipeline() = default;

    void LoadDefaultScheduler();

    void SetScheduler(std::shared_ptr<Scheduler> new_scheduler) { scheduler = new_scheduler; }

    virtual void LoadModels(const std::filesystem::path& root) = 0;

    virtual void Run(const std::string& pos_prompt, const std::string& neg_prompt, size_t steps,
                     float cfg, int image_count = 1, std::optional<uint32_t> seed = {},
                     const std::string& img = "", float denoise_strength = .5f) = 0;

protected:
    const Ort::Env* env;
    std::shared_ptr<Scheduler> scheduler;
};
