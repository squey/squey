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

#include "PVJsonSchema.h"

#include "PVJsonLines.h"

#include <arrow/type.h>
#include <arrow/util/value_parsing.h>

#include <rapidjson/document.h>

#include <QFileInfo>

#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace
{

/** The kinds of the values of a field, as bits. */
enum Kind : unsigned {
	BOOLEAN = 1 << 0,
	INTEGER = 1 << 1,
	NUMBER = 1 << 2,
	STRING = 1 << 3,
	ARRAY = 1 << 4,
	OBJECT = 1 << 5,
};

/** Reports the progress of the reading once a mebibyte more is read. */
constexpr const size_t PROGRESS_STEP = 1024 * 1024;

struct string_hash {
	using is_transparent = void;
	size_t operator()(std::string_view key) const { return std::hash<std::string_view>{}(key); }
};

/**
 * A field of the objects of the files, with the fields of its own objects.
 */
struct Field {
	std::string key;
	unsigned kinds = 0;
	/** Whether all of its strings are times. */
	bool times = true;
	/** In their order of appearance. */
	std::vector<std::unique_ptr<Field>> fields;
	std::unordered_map<std::string_view, Field*, string_hash, std::equal_to<>> fields_by_key;

	Field& child(std::string_view key)
	{
		const auto known = fields_by_key.find(key);
		if (known != fields_by_key.end()) {
			return *known->second;
		}
		fields.push_back(std::make_unique<Field>());
		Field& field = *fields.back();
		field.key = key;
		fields_by_key.emplace(field.key, &field);
		return field;
	}
};

void visit(Field& field, const rapidjson::Value& value)
{
	switch (value.GetType()) {
	case rapidjson::kNullType:
		break;
	case rapidjson::kFalseType:
	case rapidjson::kTrueType:
		field.kinds |= BOOLEAN;
		break;
	case rapidjson::kNumberType:
		field.kinds |= value.IsInt64() ? INTEGER : NUMBER;
		break;
	case rapidjson::kStringType:
		field.kinds |= STRING;
		field.times = field.times and PVRush::PVJsonSchema::microseconds_of(
		                                  {value.GetString(), value.GetStringLength()});
		break;
	case rapidjson::kArrayType:
		field.kinds |= ARRAY;
		break;
	case rapidjson::kObjectType:
		field.kinds |= OBJECT;
		// GetObj(): <windows.h> defines GetObject as a macro.
		for (const auto& member : value.GetObj()) {
			visit(field.child({member.name.GetString(), member.name.GetStringLength()}),
			      member.value);
		}
		break;
	}
}

PVRush::PVJsonSchema::Type type_of(const Field& field)
{
	using Type = PVRush::PVJsonSchema::Type;
	switch (field.kinds) {
	case BOOLEAN:
		return Type::BOOLEAN;
	case INTEGER:
		return Type::INTEGER;
	case NUMBER:
	case INTEGER | NUMBER:
		return Type::NUMBER;
	case STRING:
		return field.times ? Type::TIME : Type::TEXT;
	default:
		return Type::TEXT;
	}
}

/**
 * Appends to @p columns those of the fields of @p object, which @p fields lead to.
 */
void add_columns(const Field& object,
                 std::vector<std::string>& fields,
                 std::vector<PVRush::PVJsonSchema::Column>& columns)
{
	for (const std::unique_ptr<Field>& field : object.fields) {
		fields.push_back(field->key);
		if (field->kinds == OBJECT) {
			add_columns(*field, fields, columns);
		} else if (field->kinds != 0) {
			std::string name;
			for (const std::string& key : fields) {
				name += (name.empty() ? "" : ".") + key;
			}
			columns.push_back({fields, name, type_of(*field)});
		}
		fields.pop_back();
	}
}

std::shared_ptr<arrow::DataType> arrow_type_of(PVRush::PVJsonSchema::Type type)
{
	using Type = PVRush::PVJsonSchema::Type;
	switch (type) {
	case Type::BOOLEAN:
		return arrow::boolean();
	case Type::INTEGER:
		return arrow::int64();
	case Type::NUMBER:
		return arrow::float64();
	case Type::TIME:
		return arrow::timestamp(arrow::TimeUnit::MICRO);
	case Type::TEXT:
		break;
	}
	return arrow::utf8();
}

bool blank(const char* line, size_t length)
{
	return std::all_of(line, line + length, [](char c) { return std::isspace((unsigned char)c); });
}

} // namespace

std::shared_ptr<const PVRush::PVJsonSchema> PVRush::PVJsonSchema::infer(const QStringList& paths,
                                                                        const progress_f& progress)
{
	size_t size = 0;
	for (const QString& path : paths) {
		size += size_t(QFileInfo(path).size());
	}

	auto schema = std::make_shared<PVJsonSchema>();
	Field root;
	rapidjson::Document document;
	size_t bytes_read = 0;
	size_t reported = 0;
	for (const QString& path : paths) {
		PVJsonLines lines(path);
		while (char* line = lines.next()) {
			if (not blank(line, lines.length())) {
				// The values of the previous line are left behind before their memory is.
				document.SetNull();
				document.GetAllocator().Clear();
				document.ParseInsitu(line);
				if (document.HasParseError() or not document.IsObject()) {
					if (schema->_invalid_line_count++ == 0) {
						schema->_first_invalid_file = path;
						schema->_first_invalid_line = lines.line_number();
					}
				} else {
					visit(root, document);
					schema->_row_count++;
				}
			}
			if (progress and bytes_read + lines.bytes_read() >= reported + PROGRESS_STEP) {
				reported = bytes_read + lines.bytes_read();
				if (not progress(reported, size)) {
					return nullptr;
				}
			}
		}
		bytes_read += lines.bytes_read();
	}

	std::vector<std::string> fields;
	add_columns(root, fields, schema->_columns);
	return schema;
}

std::optional<int64_t> PVRush::PVJsonSchema::microseconds_of(std::string_view time)
{
	static const std::shared_ptr<arrow::TimestampParser> parser =
	    arrow::TimestampParser::MakeISO8601();
	int64_t value = 0;
	if ((*parser)(time.data(), time.size(), arrow::TimeUnit::MICRO, &value)) {
		return value;
	}
	// Up to nanoseconds, which have a smaller range: rounded down to microseconds.
	if ((*parser)(time.data(), time.size(), arrow::TimeUnit::NANO, &value)) {
		return value / 1000 - (value % 1000 < 0);
	}
	return std::nullopt;
}

std::shared_ptr<arrow::Schema> PVRush::PVJsonSchema::arrow_schema() const
{
	arrow::FieldVector fields;
	for (const Column& column : _columns) {
		fields.push_back(arrow::field(column.name, arrow_type_of(column.type)));
	}
	return arrow::schema(fields);
}

std::shared_ptr<const PVRush::PVJsonSchema>
PVRush::PVJsonSchema::with_columns(const QStringList& names) const
{
	auto schema = std::make_shared<PVJsonSchema>(*this);
	if (not names.isEmpty()) {
		std::erase_if(schema->_columns, [&names](const Column& column) {
			return not names.contains(QString::fromStdString(column.name));
		});
	}
	return schema;
}
