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

#include <pvcop/db/sink.h>
#include <pvcop/types/number.h>

#include <vector>

namespace PVCore
{

class PVBinaryChunk : public PVChunk
{
  public:
	PVBinaryChunk(size_t columns_count, size_t rows_count, PVRow start_index)
	    : _rows_count(rows_count), _start_index(start_index)
	{
		_columns_chunk.resize(columns_count);
	}

	template <typename T>
	void set_column_chunk(PVCol col, std::vector<T>& vec_data)
	{
		_columns_chunk[col] = pvcop::db::sink::column_chunk_t(
		    vec_data.data(), sizeof(T) * vec_data.size(),
		    std::string("number_") + pvcop::types::type_string<T>());
	}

	void set_rows_count(size_t rows_count) { _rows_count = rows_count; }

	const pvcop::db::sink::columns_chunk_t& columns_chunk() const { return _columns_chunk; }

	size_t rows_count() const override { return _rows_count; }

	void free() override { pvlogger::info() << "free PVBinaryChunk" << std::endl; /* FIXME */ };

  private:
	size_t _rows_count;
	PVRow _start_index;
	pvcop::db::sink::columns_chunk_t _columns_chunk;
};

} // namespace PVRush

#endif // __PVCORE_PVBINARYCHUNK__
