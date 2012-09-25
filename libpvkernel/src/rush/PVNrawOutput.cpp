/**
 * \file PVNrawOutput.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>

#include <tbb/parallel_invoke.h>

PVRush::PVNrawOutput::PVNrawOutput(PVRush::PVNraw &nraw_dest) :
	_nraw_dest(nraw_dest)
{
	_nraw_cur_index = 0;
}

void PVRush::PVNrawOutput::operator()(PVCore::PVChunk* out)
{
	bool ret_add;
	if (_chunk_funcs.size() > 0) {
		tbb::parallel_invoke(
			[&] {
				for (chunk_function_type const& f: this->_chunk_funcs) {
					f(out, _nraw_dest.get_number_rows());
				}
			},
			[&] {
				ret_add = this->_nraw_dest.add_chunk_utf16(*out);
			});
	}
	else {
		ret_add = _nraw_dest.add_chunk_utf16(*out);
	}

	if (ret_add) {
		// Save the chunk corresponding index
		_pvrow_chunk_idx[_nraw_cur_index] = out->agg_index();

		_nraw_cur_index++;
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

void PVRush::PVNrawOutput::job_has_finished()
{
	// Tell the destination NRAW to resize its content
	// to what it actually has, in case too much
	// elements have been pre-allocated.
	_nraw_dest.fit_to_content();
}
