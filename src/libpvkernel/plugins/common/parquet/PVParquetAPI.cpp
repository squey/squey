//
// MIT License
//
// © Squey, 2024
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

#include "PVParquetAPI.h"

#include <parquet/arrow/reader.h>
#include <parquet/schema.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/api.h>
#include <arrow/table.h>
#include <parquet/properties.h>
#include <stdint.h>
#include <boost/date_time/posix_time/posix_time_config.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <algorithm>
#include <limits>
#include <ostream>
#include <string>

#include "pvcop/db/types.h"
#include "pvkernel/rush/PVFormat_types.h"
#include "pvkernel/rush/PVXmlTreeNodeDom.h"
#include "pvlogger.h"

const std::unordered_map<arrow::Type::type, PVRush::PVParquetAPI::pvcop_type_infos>  PVRush::PVParquetAPI::pvcop_types_map = {
	{ arrow::Type::type::BOOL,				{ sizeof(pvcop::db::index_t),	"string" }},
	{ arrow::Type::type::INT8,				{ sizeof(int8_t),	"number_int8" }},
	{ arrow::Type::type::INT16,				{ sizeof(int16_t),	"number_int16" }},
	{ arrow::Type::type::INT32,				{ sizeof(int32_t),	"number_int32" }},
	{ arrow::Type::type::INT64,				{ sizeof(int64_t),	"number_int64" }},
	{ arrow::Type::type::UINT8,				{ sizeof(uint8_t),	"number_uint8" }},
	{ arrow::Type::type::UINT16,			{ sizeof(uint16_t),	"number_uint16" }},
	{ arrow::Type::type::UINT32,			{ sizeof(uint32_t),	"number_uint32" }},
	{ arrow::Type::type::UINT64,			{ sizeof(uint64_t),	"number_uint64" }},
	{ arrow::Type::type::DOUBLE,			{ sizeof(double),	"number_double" }},
	{ arrow::Type::type::FLOAT,				{ sizeof(float),	"number_float" }},
	{ arrow::Type::type::TIMESTAMP,			{ sizeof(boost::posix_time::ptime),	"time" }},
	{ arrow::Type::type::DATE32,			{ sizeof(uint64_t),	"time" }},
	{ arrow::Type::type::TIME32,			{ sizeof(boost::posix_time::time_duration),	"duration" }},
	{ arrow::Type::type::TIME64,			{ sizeof(boost::posix_time::time_duration),	"duration" }},
	{ arrow::Type::type::STRING,			{ sizeof(pvcop::db::index_t),	"string" }},
	{ arrow::Type::type::FIXED_SIZE_BINARY,	{ sizeof(pvcop::db::index_t),	"string" }},
	{ arrow::Type::type::BINARY,			{ sizeof(pvcop::db::index_t),	"string" }},
	{ arrow::Type::type::DICTIONARY,		{ sizeof(pvcop::db::index_t),	"string" }},
	{ arrow::Type::type::LIST,              { sizeof(pvcop::db::index_t),	"string" }},
    { arrow::Type::type::MAP,               { sizeof(pvcop::db::index_t),	"string" }},
    { arrow::Type::type::STRUCT,            { 0,	                        ""       }},
	// Note : "DATE64" and "DURATION" are not supported by Apache Parquet
};

constexpr const char input_column_name[] = "filename";

PVRush::PVParquetAPI::PVParquetAPI(const PVRush::PVParquetFileDescription* input_desc)
	: _input_desc(input_desc)
{
	next_file();
	get_format();
}

std::shared_ptr<arrow::Schema> PVRush::PVParquetAPI::flatten_schema(const std::shared_ptr<arrow::Schema>& schema)
{
    arrow::SchemaBuilder builder;

    std::stack<std::pair<std::shared_ptr<arrow::Field>, std::string>> field_stack;

    for (int i = schema->num_fields() - 1; i >= 0; --i) {
        field_stack.push({schema->field(i), ""});
    }

    while (!field_stack.empty()) {
        auto [field, prefix] = field_stack.top();
        field_stack.pop();

        std::string current_name = prefix.empty() ? field->name() : prefix + "." + field->name();

        if (field->type()->id() == arrow::Type::STRUCT) {
            auto struct_type = std::static_pointer_cast<arrow::StructType>(field->type());
            for (int j = struct_type->num_fields() - 1; j >= 0; --j) {
                field_stack.push({struct_type->field(j), current_name});
            }
        } else {
            (void) builder.AddField(arrow::field(
                current_name,
                field->type(),
                field->nullable(),
                field->metadata()
            ));
        }
    }

    return builder.Finish().ValueOrDie();
}

std::shared_ptr<arrow::Table> PVRush::PVParquetAPI::flatten_table(const std::shared_ptr<arrow::Table>& table)
{
    std::vector<std::shared_ptr<arrow::Array>> flat_arrays;
    std::vector<std::shared_ptr<arrow::Field>> flat_fields;
    std::stack<std::pair<std::shared_ptr<arrow::Field>, std::shared_ptr<arrow::Array>>> stack;
    for (int i = table->num_columns() - 1; i >= 0; --i) {
        const auto& col = table->column(i);
        auto array = arrow::Concatenate(col->chunks()).ValueOrDie();
        stack.push({table->schema()->field(i), array});
    }
    while (!stack.empty()) {
        auto [field, array] = stack.top();
        stack.pop();
        if (field->type()->id() == arrow::Type::STRUCT) {
            auto struct_type = std::static_pointer_cast<arrow::StructType>(field->type());
            for (int j = struct_type->num_fields() - 1; j >= 0; --j) {
                auto sub_field = struct_type->field(j);
                std::shared_ptr<arrow::Array> sub_array;
                if (array->type()->id() == arrow::Type::STRUCT) {
                    auto struct_array = std::static_pointer_cast<arrow::StructArray>(array);
                    auto struct_type = std::static_pointer_cast<arrow::StructType>(struct_array->type());
                    int idx = struct_type->GetFieldIndex(sub_field->name());
                    if (idx >= 0) {
                        sub_array = struct_array->field(idx);
                    }
                    else {
                        sub_array = std::make_shared<arrow::NullArray>(array->length());
                    }
                }
                else {
                    sub_array = std::make_shared<arrow::NullArray>(array->length());
                }
                std::string name = field->name() + "." + sub_field->name();
                stack.push({arrow::field(name, sub_field->type(), sub_field->nullable(), sub_field->metadata()), sub_array});
            }
        }
        else {
            flat_fields.push_back(field);
            flat_arrays.push_back(array);
        }
    }
    auto flat_schema = arrow::schema(flat_fields);
    return arrow::Table::Make(flat_schema, flat_arrays);
}

bool PVRush::PVParquetAPI::next_file()
{
	if (_next_input_file >= files_count()) {
		return false;
	}
	arrow::Status status = arrow::Status::OK();

	// Configure general Parquet reader settings
	auto reader_properties = parquet::ReaderProperties();
	reader_properties.set_buffer_size(4096 * 4);
	//reader_properties.enable_buffered_stream();

	// Configure Arrow-specific Parquet reader settings
	auto arrow_reader_props = parquet::ArrowReaderProperties(true /* use_threads */);
	arrow_reader_props.set_batch_size(64 * 1024); //  default is 64 * 1024
	parquet::arrow::FileReaderBuilder reader_builder;
	const std::string& parquet_file_path = _input_desc->paths()[_next_input_file].toStdString();
	status = reader_builder.OpenFile(parquet_file_path, /*memory_map=*/false, reader_properties);
	if (not status.ok()) {
		pvlogger::error() << status.ToString() << std::endl;
		return false;
	}

	{
		// expose string column dictionnaries
		std::unique_ptr<parquet::arrow::FileReader> reader;
		status = parquet::arrow::FileReader::Make(arrow::default_memory_pool(), parquet::ParquetFileReader::OpenFile(parquet_file_path), &reader);
		if (not status.ok()) {
			pvlogger::error() << status.ToString() << std::endl;
			return false;
		}
		std::shared_ptr<arrow::Schema> schema;
		status = reader->GetSchema(&schema);
		std::shared_ptr<arrow::Schema> flattened_schema = flatten_schema(schema);
		for (int i = 0; i < flattened_schema->num_fields(); ++i) {
			if (flattened_schema->field(i)->type()->id() == arrow::Type::STRING) {
				arrow_reader_props.set_read_dictionary(i, true);
			}
		}
	}

	reader_builder.properties(arrow_reader_props);
	status = reader_builder.Build(&_arrow_reader);
	if (not status.ok()) {
		pvlogger::error() << status.ToString() << std::endl;
		return false;
	}

	_next_input_file++;

	return _next_input_file < files_count();
}

void PVRush::PVParquetAPI::visit_files(const std::function<void()>& f)
{
	for (auto path : _input_desc->paths()) {
		f();
		next_file();
	}
}

QDomDocument PVRush::PVParquetAPI::get_format()
{
	if (_format == QDomDocument()) {
    	std::shared_ptr<arrow::Schema> arrow_schema;
		arrow::Status status = const_cast<PVRush::PVParquetAPI*>(this)->arrow_reader()->GetSchema(&arrow_schema);

		std::shared_ptr<arrow::Schema> flattened_schema = flatten_schema(arrow_schema);
		const arrow::FieldVector& fields = flattened_schema->fields();

		std::unique_ptr<PVXmlTreeNodeDom> format_root(PVRush::PVXmlTreeNodeDom::new_format(_format));

		if (multi_inputs()) {
			format_root->addOneField(
				input_column_name,
				"string");
		}

		for (size_t i = 0; i < fields.size(); i++) {
			std::shared_ptr<arrow::Field> field = fields[i];
			const std::shared_ptr<arrow::DataType>& data_type = field->type();
			arrow::Type::type type = data_type->storage_id();

			if (pvcop_types_map.find(type) != pvcop_types_map.end()) {
				PVRush::PVXmlTreeNodeDom* node = format_root->addOneField(
					QString::fromStdString(field->name()),
					QString::fromStdString(pvcop_types_map.at(type).string)
				);
				if (type == arrow::Type::type::TIMESTAMP) {
					const QString& time_format = "yyyy-M-d HH:m:ss.S";
					node->setAttribute(
						PVFORMAT_AXIS_TYPE_FORMAT_STR,
						time_format
					);
				}
				else if (type == arrow::Type::type::DATE32) {
					const QString& time_format = "yyyy-M-d";
					node->setAttribute(
						PVFORMAT_AXIS_TYPE_FORMAT_STR,
						time_format
					);
				}
				else if (std::string(pvcop_types_map.at(type).string) == "string") {
					QDomElement mapping = _format.createElement(PVFORMAT_AXIS_MAPPING_STR);
					mapping.setAttribute(PVFORMAT_MAP_PLOT_MODE_STR, "string");
					node->getDom().appendChild(mapping);
				}
				_column_indexes.emplace_back(i);
			}
			else {
				pvlogger::warn() << "type '"  << data_type->name() << "/" << type << "' for column '" << field->name() << "' is not supported" << std::endl;
			}
		}
	}

	return _format;
}

bool PVRush::PVParquetAPI::is_bit_optimizable() const
{
	if (multi_inputs()) {
		for (const QString& path : _input_desc->paths()) {
			std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::OpenFile(path.toStdString());
			if (parquet_reader->metadata()->num_rows() % std::numeric_limits<uint8_t>::digits != 0) {
				return false;
			}
		}
	}
	return true;
}

bool PVRush::PVParquetAPI::same_schemas() const
{
	return std::equal(_input_desc->paths().begin() +1, _input_desc->paths().end(), _input_desc->paths().begin(), [](const QString& file1, const QString& file2){
		std::unique_ptr<parquet::ParquetFileReader> reader1 = parquet::ParquetFileReader::OpenFile(file1.toStdString());
		const parquet::SchemaDescriptor* schema1 = reader1->metadata()->schema();

		std::unique_ptr<parquet::ParquetFileReader> reader2 = parquet::ParquetFileReader::OpenFile(file2.toStdString());
		const parquet::SchemaDescriptor* schema2 = reader2->metadata()->schema();

		return schema1->Equals(*schema2);
	});
}
