/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/rush/PVNraw.h> // for PVNraw
#include <pvkernel/rush/PVNrawOutput.h>

#include <pvkernel/core/PVChunk.h> // for PVChunk

#include <pvbase/types.h> // for PVRow

#include <cstddef> // for size_t
#include <map>     // for map
#include <string>  // for string

PVRush::PVNrawOutput::PVNrawOutput(PVNraw& nraw) : _nraw_dest(&nraw)
{
}

void PVRush::PVNrawOutput::operator()(PVCore::PVChunk* out)
{
	nraw_dest().add_chunk_utf16(*out);
	_out_size += out->get_init_size();

	// Clear this chunk !
	out->free();
}

PVRow PVRush::PVNrawOutput::get_rows_count()
{
	if (_nraw_dest != nullptr) {
		return _nraw_dest->get_row_count();
	} else {
		return 0;
	}
}

void PVRush::PVNrawOutput::job_has_finished(const std::map<size_t, std::string>& inv_elts)
{
	// Tell the destination NRAW that clean up can be done, everything is imported
	nraw_dest().load_done(inv_elts);
}
