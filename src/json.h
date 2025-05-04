#pragma once

#include <filesystem>
#include <map>
#include <stack>
#include <type_traits>
#include <variant>


static constexpr size_t ArrayIdx = 3;

class JsonNodeWrapper;

struct JsonNode {
	std::variant<double, std::string_view, bool, std::vector<JsonNodeWrapper>, nullptr_t> data;
	std::map<std::string_view, JsonNode> children;
};


class JsonNodeWrapper {
public:
	JsonNodeWrapper() = default;
	JsonNodeWrapper(JsonNode* node, bool owning = false) : node(node), owning(owning) {}
	JsonNodeWrapper(const JsonNodeWrapper& other) : node(other.node) {}
	~JsonNodeWrapper() { if (owning) delete node; }

	JsonNodeWrapper& operator=(const JsonNodeWrapper& other);

	JsonNodeWrapper operator[](std::string_view name) const;

	std::vector<JsonNodeWrapper>::iterator begin() const;
	std::vector<JsonNodeWrapper>::iterator end() const;

	bool Exists() const { return node != nullptr; }

	template<typename T>
	operator T() const {
		if (!node) return {};

		return std::visit([]<typename V>(const V &value) {
			if constexpr (std::convertible_to<V, T>) return static_cast<T>(value);
			else return T{};
		}, node->data);
	}

	template<typename T>
	T as(T fallback = {}) {
		if constexpr (std::is_same_v<T, std::string>) return std::string(this->as<std::string_view>(fallback));
		else return static_cast<T>(*this);
	}

	JsonNode* node = nullptr;
	bool owning = false;
};


class JsonParser {
public:
	JsonNodeWrapper Parse(const std::filesystem::path& path);

private:
	void ParseArray();
	void ParseObject();

	void ParseString();
	void ParseNumber();
	void ParseBool();
	void ParseNull();

	void ParseElement();
	void ParseNode();

	JsonNode& AddChild();
	void PushNode();

	std::vector<uint8_t> data;
	const char* ptr = nullptr;
	const char* end = nullptr;
	std::string_view object_name = "";
	std::stack<JsonNode*> nodes{};
	JsonNode root;
	int level = 0;
};
