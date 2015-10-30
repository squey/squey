/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVPureMappingProcessing.h>
#include <cassert>

PVCore::PVChunk* PVFilter::PVPureMappingProcessing::operator()(PVCore::PVChunk* chunk)
{
	if (_mappings.size() == 0) {
		return chunk;
	}

	chunk->visit_by_column(
		[&](PVRow, PVCol c, PVCore::PVField& field)
		{
			assert(c < (PVCol) _mappings.size());
			pure_mapping_func const& mf = _mappings[c];
			if (mf) {
				field.mapped_value() = mf(field);
			}
		});

	return chunk;
}
