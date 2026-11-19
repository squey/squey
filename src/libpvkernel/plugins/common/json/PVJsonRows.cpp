//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVJsonRows.h"

#include <arrow/api.h>

#include <rapidjson/writer.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace
{

struct string_hash {
	using is_transparent = void;
	size_t operator()(std::string_view key) const { return std::hash<std::string_view>{}(key); }
};

void check(const arrow::Status& status)
{
	if (not status.ok()) {
		throw std::runtime_error(status.ToString());
	}
}

} // namespace

/**
 * A field of the objects of the lines.
 */
struct PVRush::PVJsonRows::Node {
	/** The column of the values of the field, unless its objects have fields of columns. */
	std::optional<size_t> column;
	std::unordered_map<std::string, std::unique_ptr<Node>, string_hash, std::equal_to<>> fields;
};

PVRush::PVJsonRows::PVJsonRows(const PVJsonSchema& schema)
    : _schema(schema), _root(std::make_unique<Node>())
{
	const std::vector<PVJsonSchema::Column>& columns = schema.columns();
	const std::shared_ptr<arrow::Schema> arrow_schema = schema.arrow_schema();
	for (size_t i = 0; i < columns.size(); i++) {
		Node* node = _root.get();
		for (const std::string& field : columns[i].fields) {
			std::unique_ptr<Node>& child = node->fields[field];
			if (child == nullptr) {
				child = std::make_unique<Node>();
			}
			node = child.get();
		}
		node->column = i;

		std::unique_ptr<arrow::ArrayBuilder> builder;
		check(arrow::MakeBuilder(arrow::default_memory_pool(), arrow_schema->field(int(i))->type(),
		                         &builder));
		_builders.push_back(std::move(builder));
	}
}

PVRush::PVJsonRows::~PVJsonRows() = default;

bool PVRush::PVJsonRows::append(char* line)
{
	// The values of the previous line are left behind before their memory is.
	_document.SetNull();
	_document.GetAllocator().Clear();
	_document.ParseInsitu(line);
	if (_document.HasParseError() or not _document.IsObject()) {
		return false;
	}

	add_fields(*_root, _document);
	for (const std::unique_ptr<arrow::ArrayBuilder>& builder : _builders) {
		if (size_t(builder->length()) == _row_count) {
			check(builder->AppendNull());
		}
	}
	_row_count++;
	return true;
}

void PVRush::PVJsonRows::add_fields(const Node& node, const rapidjson::Value& object)
{
	// GetObj(): <windows.h> defines GetObject as a macro.
	for (const auto& member : object.GetObj()) {
		const auto field = node.fields.find(
		    std::string_view(member.name.GetString(), member.name.GetStringLength()));
		if (field == node.fields.end()) {
			continue;
		}
		const Node& child = *field->second;
		if (child.column) {
			add_value(*child.column, member.value);
		} else if (member.value.IsObject()) {
			add_fields(child, member.value);
		}
	}
}

void PVRush::PVJsonRows::add_value(size_t column, const rapidjson::Value& value)
{
	arrow::ArrayBuilder& builder = *_builders[column];
	// A null, or a value the column cannot hold, is appended as a null once the row is complete.
	if (size_t(builder.length()) > _row_count or value.IsNull()) {
		return;
	}

	switch (_schema.columns()[column].type) {
	case PVJsonSchema::Type::BOOLEAN:
		if (value.IsBool()) {
			check(static_cast<arrow::BooleanBuilder&>(builder).Append(value.GetBool()));
		}
		break;
	case PVJsonSchema::Type::INTEGER:
		if (value.IsInt64()) {
			check(static_cast<arrow::Int64Builder&>(builder).Append(value.GetInt64()));
		}
		break;
	case PVJsonSchema::Type::NUMBER:
		if (value.IsNumber()) {
			check(static_cast<arrow::DoubleBuilder&>(builder).Append(value.GetDouble()));
		}
		break;
	case PVJsonSchema::Type::TIME:
		if (value.IsString()) {
			const std::optional<int64_t> time = PVJsonSchema::microseconds_of(
			    std::string_view(value.GetString(), value.GetStringLength()));
			if (time) {
				check(static_cast<arrow::TimestampBuilder&>(builder).Append(*time));
			}
		}
		break;
	case PVJsonSchema::Type::TEXT: {
		auto& text = static_cast<arrow::StringBuilder&>(builder);
		if (value.IsString()) {
			check(text.Append(std::string_view(value.GetString(), value.GetStringLength())));
		} else {
			_text.Clear();
			rapidjson::Writer<rapidjson::StringBuffer> writer(_text);
			value.Accept(writer);
			check(text.Append(std::string_view(_text.GetString(), _text.GetSize())));
		}
		break;
	}
	}
}

arrow::ArrayVector PVRush::PVJsonRows::finish()
{
	arrow::ArrayVector arrays;
	for (const std::unique_ptr<arrow::ArrayBuilder>& builder : _builders) {
		std::shared_ptr<arrow::Array> array;
		check(builder->Finish(&array));
		arrays.push_back(std::move(array));
	}
	_row_count = 0;
	return arrays;
}
