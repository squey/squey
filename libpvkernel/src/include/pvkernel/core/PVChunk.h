/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVCORE_PVCHUNK__
#define __PVCORE_PVCHUNK__

#include "pvbase/types.h" // for chunk_index, PVCol, PVRow

namespace PVRush
{
class PVAggregator;
}

namespace PVCore
{

class PVChunk
{
	friend class PVRush::PVAggregator;

  public:
	virtual ~PVChunk() = default;

	virtual size_t rows_count() const = 0;
	virtual void set_elements_index(){};
	virtual void remove_nelts_front(size_t /*nelts_remove*/, size_t /*nstart_rem*/ = 0) {}

	virtual void free() = 0;

  protected:
	chunk_index _index = 0;
	chunk_index _agg_index = 0;
};

} // namespace PVCore

#endif // __PVCORE_PVCHUNK__