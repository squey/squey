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

#include "PVArrowRowFilter.h"

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/compute/api.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

void check(const arrow::Status& status)
{
	if (not status.ok()) {
		throw std::runtime_error(status.ToString());
	}
}

template <typename T>
T value_of(arrow::Result<T> result)
{
	check(result.status());
	return std::move(result).ValueUnsafe();
}

/**
 * A name that no column of @p schema has, for the index of the rows.
 */
std::string index_column_of(const arrow::Schema& schema)
{
	std::string name = "squey_row";
	while (not schema.GetAllFieldsByName(name).empty()) {
		name += "_";
	}
	return name;
}

/**
 * @p table, after a column of the index of each row named @p index_column. The engine cannot read
 * half floats, which become floats.
 */
std::shared_ptr<arrow::Table> indexed(const std::shared_ptr<arrow::Table>& table,
                                      const std::string& index_column)
{
	arrow::Int64Builder indexes;
	check(indexes.Reserve(table->num_rows()));
	for (int64_t row = 0; row < table->num_rows(); row++) {
		indexes.UnsafeAppend(row);
	}

	arrow::FieldVector fields{arrow::field(index_column, arrow::int64())};
	std::vector<std::shared_ptr<arrow::ChunkedArray>> columns{
	    std::make_shared<arrow::ChunkedArray>(value_of(indexes.Finish()))};
	for (int i = 0; i < table->num_columns(); i++) {
		std::shared_ptr<arrow::Field> field = table->schema()->field(i);
		std::shared_ptr<arrow::ChunkedArray> column = table->column(i);
		if (field->type()->id() == arrow::Type::HALF_FLOAT) {
			field = field->WithType(arrow::float32());
			column = value_of(arrow::compute::Cast(column, arrow::float32())).chunked_array();
		}
		fields.push_back(field);
		columns.push_back(column);
	}
	return arrow::Table::Make(arrow::schema(fields), columns, table->num_rows());
}

} // namespace

PVRush::PVArrowRowFilter::PVArrowRowFilter(const std::string& condition)
    : _filter(PVRowFilter::make(condition))
{
}

std::shared_ptr<arrow::Array>
PVRush::PVArrowRowFilter::matching_rows(const std::shared_ptr<arrow::Table>& table)
{
	const std::string index_column = index_column_of(*table->schema());
	const std::shared_ptr<arrow::Table> rows = indexed(table, index_column);
	ArrowArrayStream stream;
	check(arrow::ExportRecordBatchReader(std::make_shared<arrow::TableBatchReader>(rows), &stream));
	std::vector<int64_t> matching = _filter->matching_rows(stream, index_column);
	std::sort(matching.begin(), matching.end());

	arrow::Int64Builder indexes;
	check(indexes.AppendValues(matching));
	return value_of(indexes.Finish());
}

std::shared_ptr<arrow::Table>
PVRush::PVArrowRowFilter::filter(const std::shared_ptr<arrow::Table>& table)
{
	return value_of(arrow::compute::Take(table, matching_rows(table))).table();
}

std::string PVRush::PVArrowRowFilter::error_of(const std::string& condition,
                                               const std::shared_ptr<arrow::Schema>& schema)
{
	try {
		PVArrowRowFilter(condition).filter(value_of(arrow::Table::MakeEmpty(schema)));
	} catch (const std::exception& e) {
		return e.what();
	}
	return {};
}
