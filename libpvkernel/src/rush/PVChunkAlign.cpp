#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>

bool PVRush::PVChunkAlign::operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk)
{
	// By default, we only have one big element
	PVCore::PVElement elt(&cur_chunk, cur_chunk.begin(), cur_chunk.end());
	// That element can grow 'till the size of the chunk
	elt.set_physical_end(cur_chunk.physical_end());
	// And so can its (for now) unique field
	elt.fields().front().set_physical_end(cur_chunk.physical_end());

	cur_chunk.elements().push_back(elt);
	return true;
}

