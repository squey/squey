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

#include <pvkernel/core/PVUtils.h>

#include <algorithm>

PVRush::PVParquetSource::PVParquetSource(PVInputDescription_p input)
    : _input_desc(dynamic_cast<PVRush::PVParquetFileDescription*>(input.get()))
    , _api(_input_desc)
{
	_dicts.resize(_api.column_indexes().size());
	_dicts_ptr.resize(_dicts.size());
	for_each(_dicts.begin(), _dicts.end(), [](std::unique_ptr<pvcop::db::write_dict>& ptr) { ptr.reset(new pvcop::db::write_dict); });
	for (size_t i = 0; i < _dicts.size(); i++) {
		_dicts_ptr[i] = _dicts[i].get();
	}

	// compute source size
	const QStringList& file_paths = _input_desc->paths();
	size_t source_row_count = std::accumulate(file_paths.begin(), file_paths.end(), 0, [](size_t accumulator, const QString& file) {
		std::unique_ptr<parquet::ParquetFileReader> reader = parquet::ParquetFileReader::OpenFile(file.toStdString());
		return accumulator + reader->metadata()->num_rows();
	});
	_source_size = source_row_count * PVParquetBinaryChunk::MEGA;

	_is_bit_optimizable = _api.is_bit_optimizable();
}

PVCore::PVBinaryChunk* PVRush::PVParquetSource::operator()()
{
	bool multi_inputs = _api.multi_inputs();

	if (_recordbatch_reader == nullptr) {
		arrow::Status status = _api.arrow_reader()->GetRecordBatchReader(&_recordbatch_reader);
	}

	if (_source_start_row >= _api.row_count()) {
		if (_current_file_index == _api.files_count() - 1) {
			return nullptr;
		} else { // load next file
			_api.next_file();
			arrow::Status status = _api.arrow_reader()->GetRecordBatchReader(&_recordbatch_reader);
			_source_start_row = 0;
			_current_file_index++;
		}
	}

	std::shared_ptr<arrow::RecordBatch> record_batch = _recordbatch_reader->Next().ValueOrDie();
	const size_t record_row_count = record_batch->num_rows();

	PVRush::PVParquetBinaryChunk* chunk = new PVRush::PVParquetBinaryChunk(
		multi_inputs,
		_is_bit_optimizable,
		_current_file_index,
		record_batch.get(),
		_api.column_indexes(),
		_dicts_ptr,
		record_row_count,
		_nraw_start_row
	);

	// save column dictionaries
	if (_save_dicts) {
		_save_dicts = false;
		if (multi_inputs) { // save filename column dictionary as well
			std::unique_ptr<pvcop::db::write_dict> input_names_dict(new pvcop::db::write_dict);
			for (const QString& input_name : _input_desc->paths()) {
				input_names_dict->insert(QFileInfo(input_name).fileName().toStdString().c_str());
			}
			chunk->set_column_dict(PVCol(0), std::move(input_names_dict));
		}
		for (size_t i = 0 ; i < _api.column_indexes().size(); i++) {
			chunk->set_column_dict(PVCol(i+multi_inputs), std::move(_dicts[i]));
		}
	}

	_source_start_row += record_row_count;
	_nraw_start_row += record_row_count;

	return chunk;
}