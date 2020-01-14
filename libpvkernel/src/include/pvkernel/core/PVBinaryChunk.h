/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
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

	template <typename T>
	void set_column_chunk(PVCol col, std::vector<T>& vec_data)
	{
		_columns_chunk[col] = pvcop::db::sink::column_chunk_t(
		    vec_data.data(), vec_data.size(), sizeof(T),
		    std::string("number_") + pvcop::types::type_string<T>());
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
	std::multimap<PVCol, size_t> _invalids;
	std::vector<bool> _invalid_columns;
};

} // namespace PVCore

#endif // __PVCORE_PVBINARYCHUNK__
