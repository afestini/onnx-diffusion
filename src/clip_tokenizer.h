#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>


class ClipTokenizer {
public:
    struct PairHash {
        using is_transparent = void;

        size_t operator()(const std::pair<std::string_view, std::string_view>& p) const noexcept {
            const auto h1 = std::hash<std::string_view> {}(p.first);
            const auto h2 = std::hash<std::string_view> {}(p.second);
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
    using MergeTable = std::unordered_map<std::pair<std::string, std::string>, int, PairHash, std::equal_to<>>;

    struct VocabHash {
        using is_transparent = void;

        size_t operator()(std::string_view p) const noexcept { return std::hash<std::string_view>{}(p); }
        size_t operator()(const std::string& p) const noexcept { return std::hash<std::string>{}(p); }
        size_t operator()(const char* p) const noexcept { return std::hash<std::string_view>{}(p); }
    };
    using VocabTable = std::unordered_map<std::string, int, VocabHash, std::equal_to<>>;


    ClipTokenizer() = default;
    ClipTokenizer(const std::string& base_path);

    void Load(const std::filesystem::path& file_path);
    void LoadConfig(const std::filesystem::path& file_path);
    void LoadVocab(const std::filesystem::path& file_path);
    void LoadMerges(const std::filesystem::path& file_path);

    std::vector<int> Encode(std::string text, bool add_special_tokens = true, bool padding = true, bool truncation = true) const;
    std::string Decode(const std::vector<int>& token_ids, bool skip_special_tokens = true) const;

private:
    VocabTable vocab_to_id_;
    std::unordered_map<int, std::string> id_to_vocab_;
    MergeTable merge_ranks_;

    int max_length_ = 77;
    int unk_token_id_ = -1;
    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
    int pad_token_id_ = -1;
};
