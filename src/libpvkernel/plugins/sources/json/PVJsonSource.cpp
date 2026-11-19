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

#include "PVJsonSource.h"

#include "../parquet/PVParquetBinaryChunk.h"

#include <pvlogger.h>

#include <arrow/compute/api.h>

#include <QFileInfo>

#include <exception>
#include <stdexcept>
#include <utility>

namespace
{

/**
 * The rows of a chunk, but for the last one: a multiple of 8, for the nulls of the chunks to be
 * copied as bits into the NRAW, which a chunk starting in the middle of a byte prevents.
 */
constexpr const size_t CHUNK_ROW_COUNT = 64 * 1024;

PVRush::PVJsonFileDescription& description_of(const PVRush::PVInputDescription_p& input)
{
	return static_cast<PVRush::PVJsonFileDescription&>(*input);
}

/**
 * The schema found to make the format of the files of @p input_desc, unless the description was
 * read back, from an investigation.
 */
std::shared_ptr<const PVRush::PVJsonSchema> schema_of(PVRush::PVJsonFileDescription& input_desc)
{
	if (input_desc.schema() == nullptr) {
		input_desc.set_schema(PVRush::PVJsonSchema::infer(input_desc.paths()));
	}
	return input_desc.schema();
}

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
 * The rows [@p begin, @p end[ of @p table, in arrays of their own: the conversion of a chunk reads
 * the nulls of an array from its first row on, which a slice does not start at.
 */
std::shared_ptr<arrow::Table>
rows_of(const std::shared_ptr<arrow::Table>& table, int64_t begin, int64_t end)
{
	arrow::Int64Builder indexes;
	check(indexes.Reserve(end - begin));
	for (int64_t row = begin; row < end; row++) {
		indexes.UnsafeAppend(row);
	}
	return value_of(
	    value_of(arrow::compute::Take(table, value_of(indexes.Finish()))).table()->CombineChunks());
}

} // namespace

PVRush::PVJsonSource::PVJsonSource(PVInputDescription_p input)
    : _input(std::move(input))
    , _input_desc(description_of(_input))
    , _schema(schema_of(_input_desc))
    , _read_schema(_input_desc.row_filter().isEmpty() ? _schema->with_columns(_input_desc.columns())
                                                      : _schema)
    , _rows(*_read_schema)
{
	if (not _input_desc.row_filter().isEmpty()) {
		_row_filter = std::make_unique<PVArrowRowFilter>(_input_desc.row_filter().toStdString());
	}

	for (const QString& path : _input_desc.paths()) {
		_size += size_t(QFileInfo(path).size());
	}

	arrow::FieldVector fields = _read_schema->arrow_schema()->fields();
	if (_input_desc.multi_inputs()) {
		arrow::StringBuilder names;
		for (const QString& name : _input_desc.input_names()) {
			check(names.Append(name.toStdString()));
		}
		_input_names = value_of(names.Finish());
		fields.insert(fields.begin(),
		              arrow::field(PVParquetAPI::input_column_name,
		                           arrow::dictionary(arrow::int32(), arrow::utf8())));
	}
	_table_schema = arrow::schema(fields);

	const QStringList& columns = _input_desc.columns();
	for (int column = 0; column < int(fields.size()); column++) {
		const bool input_names = _input_names != nullptr and column == 0;
		if (input_names or columns.isEmpty() or
		    columns.contains(QString::fromStdString(fields[size_t(column)]->name()))) {
			_imported.push_back(column);
		}
	}
	for (size_t column = 0; column < _imported.size(); column++) {
		_column_indexes.push_back(column);
		_dicts.push_back(std::make_unique<pvcop::db::write_dict>());
		_dicts_ptr.push_back(_dicts.back().get());
	}
}

std::shared_ptr<arrow::Table> PVRush::PVJsonSource::read_rows()
{
	const QStringList& paths = _input_desc.paths();
	while (_rows.row_count() < CHUNK_ROW_COUNT) {
		if (_lines == nullptr) {
			if (_file_index == size_t(paths.size())) {
				break;
			}
			_lines = std::make_unique<PVJsonLines>(paths[qsizetype(_file_index)]);
		}
		char* line = _lines->next();
		if (line == nullptr) {
			_bytes_of_files_read += _lines->bytes_read();
			_lines.reset();
			_file_index++;
		} else if (_rows.append(line) and _input_names != nullptr) {
			check(_input_indexes.Append(int32_t(_file_index)));
		}
	}

	const size_t row_count = _rows.row_count();
	if (row_count == 0) {
		return nullptr;
	}

	arrow::ArrayVector arrays = _rows.finish();
	if (_input_names != nullptr) {
		arrays.insert(arrays.begin(), value_of(arrow::DictionaryArray::FromArrays(
		                                  _table_schema->field(0)->type(),
		                                  value_of(_input_indexes.Finish()), _input_names)));
	}
	return arrow::Table::Make(_table_schema, std::move(arrays), int64_t(row_count));
}

std::shared_ptr<arrow::Table> PVRush::PVJsonSource::next_rows()
{
	while (_pending_row_count < int64_t(CHUNK_ROW_COUNT) and not _read_all) {
		std::shared_ptr<arrow::Table> rows = read_rows();
		if (rows == nullptr) {
			_read_all = true;
			break;
		}
		if (_row_filter != nullptr) {
			rows = value_of(_row_filter->filter(rows)->SelectColumns(_imported));
		}
		_pending_row_count += rows->num_rows();
		_pending.push_back(std::move(rows));
	}
	if (_pending_row_count == 0) {
		return nullptr;
	}

	std::shared_ptr<arrow::Table> rows =
	    value_of(value_of(arrow::ConcatenateTables(_pending))->CombineChunks());
	_pending.clear();
	_pending_row_count = 0;
	if (rows->num_rows() > int64_t(CHUNK_ROW_COUNT)) {
		_pending.push_back(rows_of(rows, int64_t(CHUNK_ROW_COUNT), rows->num_rows()));
		_pending_row_count = _pending.back()->num_rows();
		rows = rows_of(rows, 0, int64_t(CHUNK_ROW_COUNT));
	}
	return rows;
}

PVCore::PVBinaryChunk* PVRush::PVJsonSource::operator()()
{
	std::shared_ptr<arrow::Table> rows;
	try {
		rows = next_rows();
	} catch (const std::exception& e) {
		pvlogger::error() << e.what() << std::endl;
		return nullptr;
	}
	if (rows == nullptr) {
		return nullptr;
	}

	const size_t row_count = size_t(rows->num_rows());
	auto* chunk = new PVParquetBinaryChunk(false, true, 0, rows, _column_indexes, _dicts_ptr,
	                                       row_count, _nraw_start_row);
	const size_t bytes_read = _bytes_of_files_read + (_lines ? _lines->bytes_read() : 0);
	chunk->set_init_size(bytes_read - _bytes_returned);
	_bytes_returned = bytes_read;

	if (_save_dicts) {
		_save_dicts = false;
		for (size_t column = 0; column < _dicts.size(); column++) {
			chunk->set_column_dict(PVCol(column), std::move(_dicts[column]));
		}
	}

	_nraw_start_row += row_count;
	return chunk;
}
