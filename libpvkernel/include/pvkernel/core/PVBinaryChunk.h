/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef __PVCORE_PVBINARYCHUNK__
#define __PVCORE_PVBINARYCHUNK__

#include <pvbase/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVRawSourceBase_types.h> // for EChunkType

#include <pvcop/db/sink.h>
#include <pvcop/types/number.h>

#include <vector>

namespace PVCore
{

class PVBinaryChunk : public PVChunk
{
  public:
	static constexpr PVRush::EChunkType chunk_type = PVRush::EChunkType::BINARY;

	PVBinaryChunk(size_t columns_count, size_t rows_count, PVRow start_index)
	    : _rows_count(rows_count), _start_index(start_index)
	{
		_columns_chunk.resize(columns_count);
		_invalid_columns.resize(columns_count, false);
	}

	void set_raw_column_chunk(PVCol col,
	                          const void* beg,
	                          size_t rows_count,
	                          size_t row_bytes_count,
	                          pvcop::db::type_t type)
	{
		_columns_chunk[col] =
		    pvcop::db::sink::column_chunk_t(beg, rows_count, row_bytes_count, type);
	}

	void set_column_dict(PVCol col, std::unique_ptr<pvcop::db::write_dict> dict)
	{
		_dicts.emplace_back(std::make_pair(col, std::move(dict)));
	}

	std::vector<std::pair<PVCol, std::unique_ptr<pvcop::db::write_dict>>> take_column_dicts()
	{
		return std::move(_dicts);
	}

	void set_invalid(PVCol col, size_t row) { _invalids.insert({col, row}); }
	void set_invalid_column(PVCol col) { _invalid_columns[col] = true; }

	void set_rows_count(size_t rows_count) { _rows_count = rows_count; }

	const pvcop::db::sink::columns_chunk_t& columns_chunk() const { return _columns_chunk; }

	size_t rows_count() const override { return _rows_count; }
	size_t columns_count() const { return _columns_chunk.size(); }
	size_t start_index() const { return _start_index; }
	bool is_invalid(PVCol col) const { return _invalid_columns[col]; }

	void free() override final { delete this; }

  private:
	size_t _rows_count;
	PVRow _start_index;
	pvcop::db::sink::columns_chunk_t _columns_chunk;
	std::vector<std::pair<PVCol, std::unique_ptr<pvcop::db::write_dict>>> _dicts;
	std::multimap<PVCol, size_t> _invalids;
	std::vector<bool> _invalid_columns;
};

} // namespace PVCore

#endif // __PVCORE_PVBINARYCHUNK__
