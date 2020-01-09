/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVCORE_PVERFBINARYCHUNK__
#define __PVCORE_PVERFBINARYCHUNK__

#include <pvkernel/core/PVBinaryChunk.h>
#include "../../common/erf/PVERFAPI.h"

namespace PVRush
{

class PVERFBinaryChunk : public PVCore::PVBinaryChunk
{
	static constexpr const size_t MEGA = 1024 * 1024;

  public:
	PVERFBinaryChunk(std::vector<std::vector<PVERFAPI::float_t>>&& results,
	                 size_t row_count,
	                 size_t start_index)
	    : PVCore::PVBinaryChunk(results.size(), row_count, (PVRow)start_index)
	    , _results(std::move(results))
	{
		PVCol col_count(0);
		for (const auto& result : _results) {
			set_raw_column_chunk(col_count++, (void*)(result.data()), result.size(),
			                     sizeof(PVERFAPI::float_t), PVERFAPI::float_type);
		}

		set_init_size(row_count * MEGA);
	}

  private:
	std::vector<std::vector<PVERFAPI::float_t>> _results;
};

} // namespace PVRush

#endif // __PVCORE_PVERFBINARYCHUNK__
