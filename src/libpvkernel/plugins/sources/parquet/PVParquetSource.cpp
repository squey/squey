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

#include "PVParquetSource.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <qfileinfo.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "PVParquetBinaryChunk.h"
#include "parquet/PVParquetAPI.h"
#include "../../common/arrow/PVArrowRowFilter.h"
#include <pvkernel/rush/PVFileListDescription.h>
#include "pvkernel/rush/PVInputDescription.h"

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

} // namespace

PVRush::PVParquetSource::PVParquetSource(PVInputDescription_p input)
    : _input_desc(reinterpret_cast<PVRush::PVFileListDescription*>(input.get()))
    , _api(_input_desc)
{
	_dicts.resize(_api.column_indexes().size());
	_dicts_ptr.resize(_dicts.size());
	for_each(_dicts.begin(), _dicts.end(), [](std::unique_ptr<pvcop::db::write_dict>& ptr) { ptr.reset(new pvcop::db::write_dict); });
	for (size_t i = 0; i < _dicts.size(); i++) {
		_dicts_ptr[i] = _dicts[i].get();
	}

	// The dictionary keeps a single entry per distinct name, which a file listed twice shares.
	if (_api.multi_inputs()) {
		_input_names_dict = std::make_unique<pvcop::db::write_dict>();
		for (const QString& name : _input_desc->input_names()) {
			_input_name_indexes.emplace_back(_input_names_dict->insert(name.toStdString().c_str()));
		}
	}

	// compute source size
	const QStringList& file_paths = _input_desc->paths();
	size_t source_row_count = std::accumulate(file_paths.begin(), file_paths.end(), 0, [](size_t accumulator, const QString& file) {
		std::unique_ptr<parquet::ParquetFileReader> reader = parquet::ParquetFileReader::OpenFile(file.toStdString());
		return accumulator + reader->metadata()->num_rows();
	});
	_source_size = source_row_count * PVParquetBinaryChunk::MEGA;

	_is_bit_optimizable = _api.is_bit_optimizable();

	if (not _input_desc->row_filter().isEmpty()) {
		_row_filter = std::make_unique<PVArrowRowFilter>(_input_desc->row_filter().toStdString());
		std::shared_ptr<arrow::Schema> schema;
		check(_api.arrow_reader()->GetSchema(&schema));
		_filtered_columns = PVParquetAPI::row_filter_columns(*PVParquetAPI::flatten_schema(schema));
		_input_names = _input_desc->input_names();
	}
}

std::shared_ptr<arrow::Table> PVRush::PVParquetSource::next_table()
{
	if (_recordbatch_reader == nullptr) {
		arrow::Status status = _api.arrow_reader()->GetRecordBatchReader().Value(&_recordbatch_reader);
		if (not status.ok()) {
			pvlogger::error() << status.ToString() << std::endl;
			return nullptr;
		}
	}

	if (_source_start_row >= _api.row_count()) {
		if (_current_file_index == _api.files_count() - 1) {
			return nullptr;
		} else { // load next file
			_api.next_file();
			if (_api.arrow_reader() == nullptr) { // the next file could not be opened
				return nullptr;
			}
			arrow::Status status = _api.arrow_reader()->GetRecordBatchReader().Value(&_recordbatch_reader);
			if (not status.ok()) {
				pvlogger::error() << status.ToString() << std::endl;
				return nullptr;
			}
			_source_start_row = 0;
			_current_file_index++;
		}
	}

	// A damaged file makes Next() fail, and a file holding fewer rows than its metadata
	// announces ends the stream early : neither must abort through ValueOrDie().
	std::shared_ptr<arrow::RecordBatch> record_batch;
	arrow::Status status = _recordbatch_reader->Next().Value(&record_batch);
	if (not status.ok()) {
		pvlogger::error() << status.ToString() << std::endl;
		return nullptr;
	}
	if (record_batch == nullptr) {
		return nullptr;
	}
	const size_t record_row_count = record_batch->num_rows();

	std::shared_ptr<arrow::Schema> schema;
	status = _api.arrow_reader()->GetSchema(&schema);
	if (not status.ok()) {
		pvlogger::error() << status.ToString() << std::endl;
		return nullptr;
	}
	std::shared_ptr<arrow::Table> table;
	status = arrow::Table::FromRecordBatches(schema, {record_batch}).Value(&table);
	if (not status.ok()) {
		pvlogger::error() << status.ToString() << std::endl;
		return nullptr;
	}
	_source_start_row += record_row_count;

	return _api.flatten_table(table);
}

std::shared_ptr<arrow::Table>
PVRush::PVParquetSource::kept_rows(const std::shared_ptr<arrow::Table>& flat_table)
{
	// The condition reads the columns of the types the import knows, and the names of the files.
	std::shared_ptr<arrow::Table> rows = value_of(flat_table->SelectColumns(_filtered_columns));
	if (_api.multi_inputs()) {
		const std::shared_ptr<arrow::Array> names = value_of(arrow::MakeArrayFromScalar(
		    arrow::StringScalar(_input_names[qsizetype(_current_file_index)].toStdString()),
		    rows->num_rows()));
		rows = value_of(rows->AddColumn(
		    0, arrow::field(PVParquetAPI::input_column_name, arrow::utf8()),
		    std::make_shared<arrow::ChunkedArray>(names)));
	}
	return value_of(arrow::compute::Take(flat_table, _row_filter->matching_rows(rows))).table();
}

PVCore::PVBinaryChunk* PVRush::PVParquetSource::operator()()
{
	bool multi_inputs = _api.multi_inputs();

	// The rows read, those that the row filter left out included, for the progress to count them.
	size_t read_row_count = 0;
	std::shared_ptr<arrow::Table> flat_table;
	try {
		do {
			flat_table = next_table();
			if (flat_table == nullptr) {
				return nullptr;
			}
			read_row_count += size_t(flat_table->num_rows());
			if (_row_filter != nullptr) {
				flat_table = kept_rows(flat_table);
			}
		} while (flat_table->num_rows() == 0);
	} catch (const std::exception& e) {
		pvlogger::error() << e.what() << std::endl;
		return nullptr;
	}
	const size_t record_row_count = size_t(flat_table->num_rows());

	PVRush::PVParquetBinaryChunk* chunk = new PVRush::PVParquetBinaryChunk(
		multi_inputs,
		// Filtered chunks have any count of rows, which keeps them from starting at a byte of the
		// bits of the nulls.
		_is_bit_optimizable and _row_filter == nullptr,
		multi_inputs ? _input_name_indexes[_current_file_index] : 0,
		flat_table,
		_api.column_indexes(),
		_dicts_ptr,
		record_row_count,
		_nraw_start_row
	);
	chunk->set_init_size(read_row_count * PVParquetBinaryChunk::MEGA);

	// save column dictionaries
	if (_save_dicts) {
		_save_dicts = false;
		if (multi_inputs) { // save filename column dictionary as well
			chunk->set_column_dict(PVCol(0), std::move(_input_names_dict));
		}
		for (size_t i = 0 ; i < _api.column_indexes().size(); i++) {
			chunk->set_column_dict(PVCol(i+multi_inputs), std::move(_dicts[i]));
		}
	}

	_nraw_start_row += record_row_count;

	return chunk;
}
