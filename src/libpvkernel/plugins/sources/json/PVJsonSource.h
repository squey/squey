/* * MIT License
 *
 * © Squey, 2026
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

#ifndef __PVJSONSOURCE_H__
#define __PVJSONSOURCE_H__

#include <pvkernel/core/PVBinaryChunk.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvcop/db/write_dict.h>

#include "../../common/json/PVJsonFileDescription.h"
#include "../../common/json/PVJsonLines.h"
#include "../../common/json/PVJsonRows.h"
#include "../../common/json/PVJsonSchema.h"

#include <arrow/api.h>

#include <QString>

#include <cstddef>
#include <memory>
#include <vector>

namespace PVRush
{

/**
 * Imports the JSON objects of files, one object per line, as rows of the columns of their schema,
 * after a column naming the file of each row when there are several of them.
 */
class PVJsonSource : public PVRawSourceBaseType<PVCore::PVBinaryChunk>
{
  public:
	/**
	 * Reads the files once more to find their schema, if their description has none.
	 *
	 * @throw std::runtime_error if a file cannot be read then
	 */
	explicit PVJsonSource(PVInputDescription_p input);

  public:
	QString human_name() override { return "JSON"; }
	void seek_begin() override {}
	void prepare_for_nelts(chunk_index /*nelts*/) override {}
	/** The size of the files, in bytes, as the chunks report what they read. */
	size_t get_size() const override { return _size; }
	PVCore::PVBinaryChunk* operator()() override;

  private:
	PVInputDescription_p _input;
	PVJsonFileDescription& _input_desc;
	std::shared_ptr<const PVJsonSchema> _schema;
	PVJsonRows _rows;
	size_t _size = 0;

	/** The schema of the rows, after the column of the names of the files if there are several. */
	std::shared_ptr<arrow::Schema> _table_schema;
	std::vector<size_t> _column_indexes;
	std::shared_ptr<arrow::Array> _input_names;
	arrow::Int32Builder _input_indexes;

	size_t _file_index = 0;
	std::unique_ptr<PVJsonLines> _lines;
	/** The bytes of the files read before the current one. */
	size_t _bytes_of_files_read = 0;
	/** The bytes read for the chunks returned. */
	size_t _bytes_returned = 0;
	size_t _nraw_start_row = 0;

	/** The first chunk takes them over, while all of them fill them. */
	std::vector<std::unique_ptr<pvcop::db::write_dict>> _dicts;
	std::vector<pvcop::db::write_dict*> _dicts_ptr;
	bool _save_dicts = true;
};

} // namespace PVRush

#endif // __PVJSONSOURCE_H__
