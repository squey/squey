#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVRawSourceBase.h>

void PVCore::PVChunk::init_elements_fields()
{
	PVCol nfields_src = _source->get_number_cols_to_reserve() + 2;
	PVRow nelts = _elts.size();
	allocate_fields_buffer(nelts, nfields_src);
	void* chunk_fields = _p_chunk_fields;
	list_elts::iterator it;
	size_t buffer_size_for_elt = nfields_src*sizeof(__node_list_field);
	for (it = _elts.begin(); it != _elts.end(); it++) {
		PVElement* elt = *it;
		elt->init_fields(chunk_fields, buffer_size_for_elt);
		chunk_fields = (void*) ((uintptr_t)chunk_fields + buffer_size_for_elt);
	}
}

PVCol PVCore::PVChunk::get_source_number_fields() const
{
	return _source->get_number_cols_to_reserve();
}
