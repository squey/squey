/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>

#include <tbb/parallel_invoke.h>

PVRush::PVNrawOutput::PVNrawOutput(PVNraw& nraw):
	_nraw_dest(&nraw)
{
	_nraw_cur_index = 0;
}

void PVRush::PVNrawOutput::operator()(PVCore::PVChunk* out)
{
	const bool ret_add = nraw_dest().add_chunk_utf16(*out);

	if (ret_add) {
		// Save the chunk corresponding index
		_pvrow_chunk_idx[_nraw_cur_index] = out->agg_index();

		_nraw_cur_index++;
	} else {
		// tell the pipeline it can stop
		*_stop_cond = true;
	}

	// Clear this chunk !
	out->free();
}

PVRush::PVNrawOutput::map_pvrow const& PVRush::PVNrawOutput::get_pvrow_index_map() const
{
	return _pvrow_chunk_idx;
}

void PVRush::PVNrawOutput::clear_pvrow_index_map()
{
	_nraw_cur_index = 0;
	_pvrow_chunk_idx.clear();
}

PVRow PVRush::PVNrawOutput::get_rows_count()
{
	if (_nraw_dest != nullptr) {
		return _nraw_dest->get_number_rows();
	} else {
		return 0;
	}
}

void PVRush::PVNrawOutput::job_has_finished()
{
	// Tell the destination NRAW that clean up can be done, everything is imported
	nraw_dest().load_done();
}
