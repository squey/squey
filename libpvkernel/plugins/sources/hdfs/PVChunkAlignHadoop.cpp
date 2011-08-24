#include "PVChunkAlignHadoop.h"
#include "PVInputHadoop.h"


PVRush::PVChunkAlignHadoop::PVChunkAlignHadoop(PVInputHadoop& input, PVCol nfields):
	_input(input), _conv_buf(NULL)
{
	// Hadoop gives us UTF8 data
	// Use ICU to convert this back to UTF16 !
	UErrorCode status = U_ZERO_ERROR;
	_ucnv = ucnv_open("UTF-8", &status);
	ucnv_resetToUnicode(_ucnv);
	_nfields = nfields;
}

PVRush::PVChunkAlignHadoop::~PVChunkAlignHadoop()
{
	if (_conv_buf) {
		free(_conv_buf);
	}
}

bool PVRush::PVChunkAlignHadoop::operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk)
{
	// An hadoop NRAW element is transmitted like this:
	// [source_offset: 8 bytes] [element_size: 4 bytes] [ [field_size: 4 bytes] [..field content..] ...]
	if (_conv_buf == NULL) {
		_conv_buf = (char*) malloc(cur_chunk.size()+cur_chunk.avail());
	}
	
	// Pointer to the last offset of the last complete element
	offset_type* last_elt = NULL;
	offset_type* last_elt_end = NULL;

	char* cur = _conv_buf;
	memcpy(cur, cur_chunk.begin(), cur_chunk.size());
	char* end = cur + cur_chunk.size();
	char* cur_utf16 = cur_chunk.begin();
	char* end_utf16 = cur_utf16 + cur_chunk.size() + cur_chunk.avail();
	UErrorCode status = U_ZERO_ERROR;
	while (cur < end) {
		if ((uintptr_t)(end)-(uintptr_t)cur < sizeof(offset_type)+sizeof(element_length_type)) {
			break;
		}

		// Get current offset and element length
		offset_type* off = (offset_type*) cur;
		cur += sizeof(offset_type);
		element_length_type elt_length = *((element_length_type*) cur);
		cur += sizeof(element_length_type);
		last_elt = off;
		if ((uintptr_t)(end)-(uintptr_t)(cur) < elt_length) {
			break;
		}
		char* end_elt = cur+elt_length;

		// Now concert to UTF16 and create the fields.
		PVCore::PVElement* elt = cur_chunk.add_element(cur_utf16, cur_utf16);
		elt->fields().clear();
		char* cur_field = cur;
		for (PVCol j = 0; j < _nfields; j++) {
			field_length_type field_length = *((field_length_type*) cur_field);
			cur_field += sizeof(field_length_type);

			if ((uintptr_t)end-(uintptr_t)cur_field < field_length) {
				// Invalid element, discard it.
				elt->set_invalid();
				PVLOG_ERROR("(PVChunkAlignHadoop::operator()) field length '%u' is too large !\n", field_length);
				break;
			}

			// Do not use ucnv_toUnicode as we are doing no "streaming" conversion, but only "full string" conversions.
			int32_t sconv = ucnv_toUChars(_ucnv, (UChar*) cur_utf16, ((uintptr_t)end_utf16-(uintptr_t)cur_utf16)/sizeof(UChar), cur_field, field_length, &status);
			if (status == U_BUFFER_OVERFLOW_ERROR) {
				// That should not happen !
				PVLOG_ERROR("(PVChunkAlignHadoop::operator()) size of chunk is too small to handle UTF16 conversion !\n");
				elt->set_invalid();
				break;
			}
			char* end_field_utf16 = cur_utf16 + sconv*sizeof(UChar);
			PVCore::PVField f(*elt, cur_utf16, end_field_utf16);
			elt->fields().push_back(f);
			cur_utf16 = end_field_utf16;
			cur_field += field_length;
		}
		if (end_elt != cur_field) {
			PVLOG_WARN("(PVChunkAlignHadoop::operator()) end of hadoop element is at '%x', and last field finished at '%x'.\n", end_elt, cur_field);
		}
		elt->set_physical_end(cur_utf16);
		elt->set_end(cur_utf16);

		last_elt_end = (offset_type*) end_elt;
		cur = end_elt;
	}

	if (cur_chunk.c_elements().size() == 0) {
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
