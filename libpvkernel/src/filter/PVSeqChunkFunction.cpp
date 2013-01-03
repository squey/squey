#include <pvkernel/filter/PVSeqChunkFunction.h>



PVCore::PVChunk* PVFilter::PVSeqChunkFunction::operator()(PVCore::PVChunk* chunk)
{
	for (chunk_function_type const& f: this->_chunk_funcs) {
		f(chunk, _cur_row);
	}
	_cur_row += chunk->get_nelts_valid();

	return chunk;
}
