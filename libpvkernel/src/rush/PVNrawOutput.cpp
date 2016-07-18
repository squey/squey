/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/rush/PVNrawOutput.h>

PVRush::PVNrawOutput::PVNrawOutput(PVNraw& nraw) : _nraw_dest(&nraw)
{
}

void PVRush::PVNrawOutput::operator()(PVCore::PVChunk* out)
{
	nraw_dest().add_chunk_utf16(*out);

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

void PVRush::PVNrawOutput::job_has_finished()
{
	// Tell the destination NRAW that clean up can be done, everything is imported
	nraw_dest().load_done();
}
