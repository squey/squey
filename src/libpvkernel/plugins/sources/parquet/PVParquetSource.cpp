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
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <qfileinfo.h>
#include <QDir>
#include <QSet>
#include <algorithm>
#include <numeric>
#include <utility>

#include "PVParquetBinaryChunk.h"
#include "parquet/PVParquetAPI.h"
#include "parquet/PVParquetFileDescription.h"
#include "pvkernel/rush/PVInputDescription.h"

namespace
{

/**
 * The names of the files of @p paths: their file names, or, when two of them share one, their
 * paths from the deepest directory holding them all.
 */
QStringList input_names(const QStringList& paths)
{
	QStringList names;
	for (const QString& path : paths) {
		names << QFileInfo(path).fileName();
	}
	if (QSet<QString>(names.begin(), names.end()).size() == names.size()) {
		return names;
	}

	QStringList absolute_paths;
	for (const QString& path : paths) {
		absolute_paths << QDir::cleanPath(QFileInfo(path).absoluteFilePath());
	}

	QStringList common_dirs = QFileInfo(absolute_paths.front()).path().split('/');
	for (const QString& path : absolute_paths) {
		const QStringList dirs = QFileInfo(path).path().split('/');
		qsizetype shared = 0;
		while (shared < common_dirs.size() and shared < dirs.size() and
		       common_dirs[shared] == dirs[shared]) {
			shared++;
		}
		common_dirs.resize(shared);
	}
	// Files on different Windows drives have no directory in common.
	if (common_dirs.isEmpty()) {
		return absolute_paths;
	}
	// An absolute path starts with an empty name, before its first separator.
	const QDir common_dir(common_dirs == QStringList{""} ? "/" : common_dirs.join('/'));

	names.clear();
	for (const QString& path : absolute_paths) {
		names << common_dir.relativeFilePath(path);
	}
	return names;
}

} // namespace

PVRush::PVParquetSource::PVParquetSource(PVInputDescription_p input)
    : _input_desc(reinterpret_cast<PVRush::PVParquetFileDescription*>(input.get()))
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
		for (const QString& name : input_names(_input_desc->paths())) {
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
}

PVCore::PVBinaryChunk* PVRush::PVParquetSource::operator()()
{
	bool multi_inputs = _api.multi_inputs();

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
	std::shared_ptr<arrow::Table> flat_table = _api.flatten_table(table);

	PVRush::PVParquetBinaryChunk* chunk = new PVRush::PVParquetBinaryChunk(
		multi_inputs,
		_is_bit_optimizable,
		multi_inputs ? _input_name_indexes[_current_file_index] : 0,
		flat_table,
		_api.column_indexes(),
		_dicts_ptr,
		record_row_count,
		_nraw_start_row
	);

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

	_source_start_row += record_row_count;
	_nraw_start_row += record_row_count;

	return chunk;
}
