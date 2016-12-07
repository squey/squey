/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVRawSourceBase.h> // for PVRawSourceBase

#include "pvkernel/core/PVElement.h" // for PVElement
#include <pvkernel/core/PVChunk.h>

#include "pvbase/types.h" // for PVCol, PVRow

#include <cstddef> // for size_t
#include <cstdint> // for uintptr_t

void PVCore::PVChunk::init_elements_fields()
{
	PVCol nfields_src = _source->get_number_cols_to_reserve() + PVCol(2);
	PVRow nelts = _elts.size();
	allocate_fields_buffer(nelts, nfields_src);
	void* chunk_fields = _p_chunk_fields;
	size_t buffer_size_for_elt = nfields_src * sizeof(__node_list_field);
	for (PVElement* elt : _elts) {
		elt->init_fields(chunk_fields, buffer_size_for_elt);
		chunk_fields = (void*)((uintptr_t)chunk_fields + buffer_size_for_elt);
	}
}
