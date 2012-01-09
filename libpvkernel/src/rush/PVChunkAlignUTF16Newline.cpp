#include <pvkernel/rush/PVChunkAlignUTF16Newline.h>
#include <pvkernel/core/PVField.h>

PVRush::PVChunkAlignUTF16Newline::PVChunkAlignUTF16Newline() :
	_align_char(QChar('\n'))
{
}


bool PVRush::PVChunkAlignUTF16Newline::operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk)
{
	if (!_align_char(cur_chunk, next_chunk)) {
		return false;
	}

	PVCore::list_elts::iterator it;
	for (it = cur_chunk.elements().begin(); it != cur_chunk.elements().end(); it++) {
		PVCore::PVElement &elt = *(*it);
		if(elt.size() > sizeof(QChar)) {
			char* last_char = elt.end()-sizeof(QChar);
			// Check if the last character is '\r'
			if (*((uint16_t*)(last_char)) == 0x000d) {
				elt.set_end(last_char);
				//elt.fields().front().set_end(last_char);
			}
		}
	}

	return true;
}
