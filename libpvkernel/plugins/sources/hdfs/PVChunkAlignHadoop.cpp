#include "PVChunkAlignHadoop.h"
#include "PVInputHadoop.h"


PVRush::PVChunkAlignHadoop::PVChunkAlignHadoop(PVInputHadoop& input):
	_input(input)
{
}

bool PVRush::PVChunkAlignHadoop::operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk)
{
	// An hadoop NRAW element is transmitted like this:
	// [source_offset: 8 bytes] [element_size: 4 bytes] [ [field_size: 4 bytes] [..field content..] ...]
	
	// Pointer to the last offset of the last complete element
	offset_type* last_elt = NULL;
	offset_type* last_elt_end = NULL;
	char* cur = cur_chunk.begin();
	char* end = cur_chunk.end();

	while (cur < end) {
		if ((uintptr_t)(end)-(uintptr_t)cur < sizeof(offset_type)+sizeof(element_length_type)) {
			break;
		}

		// Get current offset and element length
		offset_type* off = (offset_type*) cur;
		cur += sizeof(offset_type);
		element_length_type* elt_length = (element_length_type*) cur;
		cur += sizeof(element_length_type);
		if ((uintptr_t)(end)-(uintptr_t)(cur) < *elt_length) {
			break;
		}
		last_elt = off;
		char* end_elt = cur+*elt_length;
		cur_chunk.add_element(cur, end_elt);
		cur = end_elt;
		last_elt_end = (offset_type*) end_elt;
	}

	if (last_elt == NULL) {
		// Chunk is too small, we couldn't get an element !
		return false;
	}

	// Tell the "last seen offset" to the input source
	_input.set_last_seen_offset(*last_elt);

	// Move what's remaining to the next chunk
	uintptr_t sremains = (uintptr_t)end-(uintptr_t)last_elt_end;
	memcpy(next_chunk.begin(), (char*) last_elt_end, sremains); 
	next_chunk.set_end(next_chunk.begin()+sremains);

	return true;
}
