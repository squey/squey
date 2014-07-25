/**
 * \file PVNrawOutput.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>

#include <tbb/parallel_invoke.h>

PVRush::PVNrawOutput::PVNrawOutput():
	_nraw_dest(nullptr)
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
	// Tell the destination NRAW to resize its content
	// to what it actually has, in case too much
	// elements have been pre-allocated.
	nraw_dest().fit_to_content();
}
