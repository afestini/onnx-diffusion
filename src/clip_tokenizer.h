#include <iostream>
#include <ranges>
#include <string>
#include <unordered_map>


consteval bool is_printable(uint8_t b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

struct UTF8Byte {
    explicit (false) operator string_view() const { return string_view(codepoint.data(), size); }
    array<char, 4> codepoint;
    uint8_t size = 0;
};

consteval UTF8Byte utf8_encode(uint32_t cp) {
    if (cp <= 0x7F) {
        return { {
            static_cast<char>(cp) },
            1
        };
    }
    else if (cp <= 0x7FF) {
        return { {
            static_cast<char>(0xC0 | (cp >> 6)),
            static_cast<char>(0x80 | (cp & 0x3F)) },
            2
        };
    }
    else if (cp <= 0xFFFF) {
        return { {
            static_cast<char>(0xE0 | (cp >> 12)),
            static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
            static_cast<char>(0x80 | (cp & 0x3F)) },
            3
        };
    }
    else {
        return { {
            static_cast<char>(0xF0 | (cp >> 18)),
            static_cast<char>(0x80 | ((cp >> 12) & 0x3F)),
            static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
            static_cast<char>(0x80 | (cp & 0x3F)) },
            4
        };
    }
}


consteval array<UTF8Byte, 256> make_byte_encoder() {
    array<UTF8Byte, 256> encoder {};
    for (int b = 0, extra = 0; b < 256; ++b) {
        const auto printable = is_printable(static_cast<uint8_t>(b));
        encoder[b] = printable ? utf8_encode(b) : utf8_encode(256u + extra++);
    }
    return encoder;
}
static inline constexpr auto byte_to_unicode = make_byte_encoder();


static unordered_map<string_view, uint8_t> make_byte_decoder() {
    unordered_map<string_view, uint8_t> decoder;
    for (auto [i, str] : views::enumerate(byte_to_unicode))
        decoder[str] = static_cast<uint8_t>(i);
    return decoder;
}
static inline const auto unicode_to_byte = make_byte_decoder();



class ClipTokenizer {
public:
    struct PairHash {
        using is_transparent = void;

        size_t operator()(const pair<string_view, string_view>& p) const noexcept {
            const auto h1 = hash<string_view> {}(p.first);
            const auto h2 = hash<string_view> {}(p.second);
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
    using MergeTable = unordered_map<pair<string, string>, int, PairHash, equal_to<>>;

    struct VocabHash {
        using is_transparent = void;

        size_t operator()(string_view p) const noexcept { return hash<string_view>{}(p); }
        size_t operator()(const string& p) const noexcept { return hash<string>{}(p); }
        size_t operator()(const char* p) const noexcept { return hash<string_view>{}(p); }
    };


    ClipTokenizer(const string& vocab_filepath, const string& merges_filepath) {
        load_vocab_json(vocab_filepath);

        load_merges(merges_filepath);

        // TODO load from config
        // Get IDs for special tokens
        if (const auto it = vocab_to_id_.find("<|endoftext|>"sv); it != vocab_to_id_.end()) {
            eos_token_id_ = it->second;
            unk_token_id_ = eos_token_id_; // CLIP often uses EOS for UNK
        }
        if (const auto it = vocab_to_id_.find("<|startoftext|>"sv); it != vocab_to_id_.end()) {
            bos_token_id_ = it->second;
        }
        // Assuming padding token is the same as EOS token
        pad_token_id_ = eos_token_id_;
    }


    void load_vocab_json(const string& file_path) {
        const auto json_string = read_file(file_path);

        for (const auto line_sr : views::split(json_string, '\n')) {
            const string_view line(line_sr);
            const size_t start = line.find('"');
            const size_t end = line.rfind('"');

            if (start == end) continue;

            const auto key = line.substr(start + 1, end - start - 1);

            const size_t colon = line.find(':', end + 1);
            if (colon == string_view::npos) continue;

            size_t num_start = line.find_first_not_of(" \t", colon + 1);
            if (num_start == string_view::npos) continue;

            int value = 0;
            from_chars(line.data() + num_start, to_address(line.end()), value);

            vocab_to_id_.emplace(key, value);
            id_to_vocab_.emplace(value, key);
        }
    }


    void load_merges(const string& file_path) {
        const auto merges_string = read_file(file_path);

        int rank = 0;
        const auto to_drop = merges_string.starts_with("#version: ") ? 1 : 0;
        for (auto line : views::split(merges_string, '\n') | views::drop(to_drop)) {
            const auto entry = string_view(line);
            const auto pos = entry.find(' ');
            if (pos == string::npos)
                continue;

            const auto token1 = entry.substr(0, pos);
            const auto token2 = entry.substr(pos + 1);

            if (!token1.empty() && !token2.empty())
                merge_ranks_.emplace(pair { token1, token2 }, rank++);
        }
    }


    vector<int> encode(string text, int max_length, bool add_special_tokens = true, bool padding = true, bool truncation = true) const {
        ranges::transform(text, text.data(), [](char c){ return tolower(c); });

        static constexpr array special_tokens { "<|startoftext|>"sv, "<|endoftext|>"sv, "'s"sv, "'t"sv, "'re"sv, "'ve"sv, "'m"sv, "'ll"sv, "'d"sv };
 
        vector<string_view> initial_tokens;
        string_view input(text);
        while (!input.empty()) {
            const unsigned char ch = static_cast<unsigned char>(input[0]);
            if (isspace(ch)) {
                input.remove_prefix(1);
            }
            else if (const auto it = ranges::find_if(special_tokens, [&input](auto tok) { return input.starts_with(tok); }); it != special_tokens.end()) {
                initial_tokens.emplace_back(*it);
                input.remove_prefix(it->length());
            }
            else if (isdigit(ch)) {
                initial_tokens.emplace_back(input.substr(0, 1));
                input.remove_prefix(1);
            }
            else if (isalpha(ch)) {
                const auto len = ranges::distance(input.begin(), ranges::find_if_not(input, [](unsigned char c) {return isalpha(c); }));
                initial_tokens.emplace_back(input.substr(0, len));
                input.remove_prefix(len);
            }
            else {
                const auto len = ranges::distance(input.begin(), ranges::find_if(input, [](unsigned char c) {return isspace(c) || isdigit(c) || isalpha(c); }));
                initial_tokens.emplace_back(input.substr(0, len));
                input.remove_prefix(len);
            }
        }

        // 2. Byte Encoding and Combine Segments into a single sequence of characters
        string token_string;
        token_string.reserve(text.size() * 2);
        vector<string_view> bpe_tokens;
        bpe_tokens.reserve(text.size() * 2);

        size_t offset = 0;
        for (const auto& token : initial_tokens) {
            for (unsigned char byte : token) {
                offset = token_string.size();
                token_string += byte_to_unicode[byte];
                bpe_tokens.emplace_back(token_string.begin() + offset, token_string.end());
            }
            token_string.append("</w>");
            bpe_tokens.back() = string_view(token_string.begin() + offset, token_string.end());
        }

        // 3. Perform BPE Merges across the entire sequence of byte-encoded characters/tokens
        // This loop continues as long as merges are found.
        for (;;) {
            const pair<string, string>* best_pair = nullptr;
            int best_rank = numeric_limits<int>::max();
            size_t merge_idx = 0;

            // Find the best pair to merge based on merge ranks across the entire current sequence
            for (const auto& current_pair : views::adjacent<2>(bpe_tokens)) {
                if (const auto it = merge_ranks_.find(current_pair); it != merge_ranks_.end() && it->second < best_rank) {
                    best_pair = &it->first;
                    best_rank = it->second;
                    merge_idx = &get<0>(current_pair) - bpe_tokens.data();
                }
            }

            if (!best_pair) break;

            // Perform the merge: Replace all occurrences of the best_pair with the merged token and update in-place
            for (size_t i = merge_idx; i < bpe_tokens.size(); ++i, ++merge_idx) {
                if (bpe_tokens[i] == best_pair->first && bpe_tokens[i + 1] == best_pair->second) {
                    bpe_tokens[merge_idx] = string_view(bpe_tokens[i].data(), best_pair->first.size() + best_pair->second.size());
                    ++i;
                }
                else
                    bpe_tokens[merge_idx] = bpe_tokens[i];
            }
            bpe_tokens.resize(merge_idx);
        }

        // 4. Convert final BPE tokens to IDs and handle special tokens
        vector<int> token_ids;
        token_ids.reserve(max_length);

        if (add_special_tokens && bos_token_id_ != -1)
            token_ids.push_back(bos_token_id_);

        for (const string_view token : bpe_tokens) {
            if (const auto it = vocab_to_id_.find(token); it != vocab_to_id_.end())
                token_ids.push_back(it->second);
            else if (unk_token_id_ != -1)
                token_ids.push_back(unk_token_id_);
            else
                cerr << "Warning: Token '" << token << "' not found in vocabulary and no UNK token defined. Skipping." << endl;
        }

        if (add_special_tokens && eos_token_id_ != -1)
            token_ids.push_back(eos_token_id_);

        // 5. Truncation
        if (truncation && token_ids.size() > max_length)
            token_ids.resize(max_length);

        // 6. Padding
        if (padding && token_ids.size() < max_length)
            token_ids.insert(token_ids.end(), max_length - token_ids.size(), pad_token_id_);

        return token_ids;
    }


    string decode(const vector<int>& token_ids, bool skip_special_tokens = true) const {
        string decoded_text = "";

        for (int id : token_ids) {
            if (skip_special_tokens && (id == bos_token_id_ || id == eos_token_id_ || id == unk_token_id_ || id == pad_token_id_)) {
                continue;
            }
            if (const auto it = id_to_vocab_.find(id); it != id_to_vocab_.end()) {
                // Decode from byte-encoded token back to original bytes
                if (const auto uni_it = unicode_to_byte.find(it->second); uni_it != unicode_to_byte.end()) {
                    decoded_text += static_cast<unsigned char>(uni_it->second);
                }
                else {
                    // If a character is not in the byte_encoder_ mapping, keep it as is.
                    decoded_text += it->second;
                }

            }
            else {
                cerr << "Warning: Token ID '" << id << "' not found in vocabulary during decoding. Skipping." << endl;
                // Optionally add a placeholder for unknown IDs
                // decoded_text += "<UNK>";
            }
        }

        // Post-processing for decoding often involves handling spaces that were part of tokens.
        // A simple approach is to replace the special space character introduced by BPE with a regular space.
        // This part might need fine-tuning based on the specific CLIP tokenizer's decoding behavior.
        // The character U+0120 (?) is often used to represent spaces in BPE.
        const UTF8Byte space_char = byte_to_unicode.at(32);
        size_t pos = decoded_text.find(space_char);
        while (pos != string::npos) {
            decoded_text.replace(pos, 1, " ");
            pos = decoded_text.find(space_char, pos + 1);
        }

        return decoded_text;
    }

private:
    unordered_map<string, int, VocabHash, equal_to<>> vocab_to_id_;
    unordered_map<int, string> id_to_vocab_;
    MergeTable merge_ranks_;

    int unk_token_id_ = -1;
    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
    int pad_token_id_ = -1;
};
