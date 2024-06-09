/* * MIT License
 *
 * © Squey, 2024
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVParquetSOURCE_FILE_H__
#define __PVParquetSOURCE_FILE_H__

#include "PVParquetSource.h"

#include <fcntl.h>
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/core/serialize_numbers.h>
#include <arrow/record_batch.h>
#include <qcontainerfwd.h>
#include <qlist.h>
#include <stddef.h>
#include <iterator>
#include <memory>
#include <QString>
#include <vector>

#include "../../common/parquet/PVParquetFileDescription.h"
#include "../../common/parquet/PVParquetAPI.h"
#include "PVParquetSource.h"
#include "PVParquetBinaryChunk.h"
#include "pvbase/types.h"
#include "pvcop/db/write_dict.h"
#include "pvkernel/core/PVBinaryChunk.h"

namespace PVRush
{
class PVParquetFileDescription;

class PVParquetSource : public PVRawSourceBaseType<PVCore::PVBinaryChunk>
{
  public:
  	using string_dicts = std::vector<std::unique_ptr<pvcop::db::write_dict>>;
	using string_dicts_ptr = std::vector<pvcop::db::write_dict*>;

  private:
	static constexpr const size_t MEGA = 1024 * 1024;

  public:
	PVParquetSource(PVInputDescription_p input);

	QString human_name() override { return "Parquet"; }
	void seek_begin() override {}
	void prepare_for_nelts(chunk_index /*nelts*/) override {}
	size_t get_size() const override { return _source_size; }
	PVCore::PVBinaryChunk* operator()() override;

  private:
	PVRush::PVParquetFileDescription* _input_desc;
	PVRush::PVParquetAPI _api;
	std::unique_ptr<::arrow::RecordBatchReader> _recordbatch_reader;

	QStringList _files_path;
	bool _is_bit_optimizable = false;

	// source
	size_t _source_start_row = 0;
	size_t _nraw_start_row = 0;
	size_t _current_file_index = 0;
	size_t _source_size = 0;

	string_dicts _dicts;
	string_dicts_ptr _dicts_ptr;
	bool _save_dicts = true;

};

} // namespace PVRush

#endif // __PVParquetSOURCE_FILE_H__
