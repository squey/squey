/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/rush/PVNraw.h> // for PVNraw
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVFormat.h>

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/PVBinaryChunk.h>

#include <pvbase/types.h> // for PVRow

#include <cstddef> // for size_t
#include <map>     // for map
#include <string>  // for string

PVRush::PVNrawOutput::PVNrawOutput(PVNraw& nraw) : _nraw_dest(&nraw) {}

void PVRush::PVNrawOutput::operator()(PVCore::PVChunk* out)
{

	if (PVCore::PVTextChunk* text_chunk = dynamic_cast<PVCore::PVTextChunk*>(out)) {
		nraw_dest().add_chunk_utf16(*text_chunk);
	} else {
		assert(dynamic_cast<PVCore::PVBinaryChunk*>(out));
		PVCore::PVBinaryChunk* bin_chunk = static_cast<PVCore::PVBinaryChunk*>(out);

		nraw_dest().add_bin_chunk(*bin_chunk);
	}

	_out_size += out->get_init_size();

	// Clear this chunk !
	out->free();
}

PVRow PVRush::PVNrawOutput::get_rows_count()
{
	if (_nraw_dest != nullptr) {
		return _nraw_dest->row_count();
	} else {
		return 0;
	}
}

void PVRush::PVNrawOutput::prepare_load(const PVRush::PVFormat& format)
{
	_nraw_dest->prepare_load(format.get_storage_format());
}

void PVRush::PVNrawOutput::job_has_finished(const std::map<size_t, std::string>& inv_elts)
{
	// Tell the destination NRAW that clean up can be done, everything is imported
	nraw_dest().load_done(inv_elts);
}
