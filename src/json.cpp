#include <algorithm>
#include <vector>

#include "json.h"

using namespace std;


JsonNodeWrapper& JsonNodeWrapper::operator=(const JsonNodeWrapper& other) {
	if (owning) delete node;
	node = other.node;
	owning = false;
	return *this;
}


JsonNodeWrapper JsonNodeWrapper::operator[](string_view name) const {
	if (!node) return {};

	if (const auto it = node->children.find(name); it != node->children.end()) return {&it->second};
	return {};
}


vector<JsonNodeWrapper>::iterator JsonNodeWrapper::begin() const {
	const auto arr = node ? get_if<ArrayIdx>(&node->data) : nullptr;
	return arr ? arr->begin() : vector<JsonNodeWrapper>::iterator{};
}


vector<JsonNodeWrapper>::iterator JsonNodeWrapper::end() const {
	const auto arr = node ? get_if<ArrayIdx>(&node->data) : nullptr;
	return arr ? arr->end() : vector<JsonNodeWrapper>::iterator{};
}




JsonNodeWrapper JsonParser::Parse(const filesystem::path& path) {
	auto file = fopen(path.string().c_str(), "rb");
	if (!file) return {};

	fpos_t len {};
	fseek(file, 0, SEEK_END);
	fgetpos(file, &len);
	fseek(file, 0, SEEK_SET);

	data.resize(len);
	fread(data.data(), 1, data.size(), file);
	fclose(file);

	ptr = bit_cast<const char*>(data.data());
	end = bit_cast<const char*>(to_address(data.end()));

	ParseElement();

	return {&root};
}


void JsonParser::ParseArray() {
	PushNode();
	nodes.top()->data.emplace<ArrayIdx>();
	ParseNode();
}


void JsonParser::ParseObject() {
	PushNode();
	ParseNode();
}


void JsonParser::ParseString() {
	if (ptr >= end) return;

	const char* from = ptr;
	while (ptr < end) {
		const char sym = *ptr++;
		if (sym == '"') break;
		if (sym == '\\') ptr = min(ptr + 1, end);
	}

	string_view str(from, ptr - from - 1);

	if (!object_name.empty() || nodes.top()->data.index() == ArrayIdx)
		AddChild().data = str;
	else
		object_name = str;
}


void JsonParser::ParseNumber() {
	static constexpr string_view delim = " ,\n\t\r]}";

	const char* from = --ptr;
	while (!ranges::contains(delim, *ptr)) ++ptr;

	if (ptr != from) {
		const double value = stod(string(from, ptr - from));
		AddChild().data = value;
	}
}


void JsonParser::ParseBool() {
	bool b = false;

	if (string_view(ptr - 1, 4) == "true") {
		b = true;
		ptr += 3;
	}
	else if (string_view(ptr - 1, 5) == "false") {
		ptr += 4;
	}
	else return;

	AddChild().data = b;
}


void JsonParser::ParseNull() {
	if (string_view(ptr - 1, 4) != "null") return;

	ptr += 3;
	AddChild().data = nullptr;
}


void JsonParser::ParseElement() {
	while (ptr < end) {
		const char symbol = *ptr++;

		switch (symbol) {
		case ' ':
		case '\t':
		case '\n':
		case '\r':
		case ',': continue;
		case '[': ParseArray(); break;
		case ']': --level; nodes.pop(); return;
		case '{': ParseObject(); break;
		case '}': --level; nodes.pop(); return;
		case '"': ParseString(); break;
		case 't':
		case 'f': ParseBool(); break;
		case 'n': ParseNull(); break;
		case ':': break;
		default: ParseNumber(); break;
		}
	}
}


void JsonParser::ParseNode() {
	int element_level = level++;
	while (level > element_level) {
		ParseElement();
	}
}


JsonNode& JsonParser::AddChild() {
	if (nodes.top()->data.index() == ArrayIdx)
		return *get<ArrayIdx>(nodes.top()->data).emplace_back(new JsonNode, true).node;
	
	auto& child = nodes.top()->children.try_emplace(object_name).first->second;
	object_name = "";
	return child;
}


void JsonParser::PushNode() {
	if (!nodes.empty())
		nodes.push(&AddChild());
	else
		nodes.push(&root);
}
