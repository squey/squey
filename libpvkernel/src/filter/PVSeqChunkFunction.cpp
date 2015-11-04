/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVSeqChunkFunction.h>



PVCore::PVChunk* PVFilter::PVSeqChunkFunction::operator()(PVCore::PVChunk* chunk)
{
	for (chunk_function_type const& f: this->_chunk_funcs) {
		f(chunk, _cur_row);
	}
	_cur_row += chunk->get_nelts_valid();

	return chunk;
}
