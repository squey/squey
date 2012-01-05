#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>

#include <tbb/tbb_allocator.h>

PVRush::PVNrawOutput::PVNrawOutput(PVRush::PVNraw &nraw_dest) :
	_nraw_dest(nraw_dest)
{
	_nraw_cur_index = 0;
}

void PVRush::PVNrawOutput::operator()(PVCore::PVChunk* out)
{
	// Write all elements of the chunk in the final nraw
	PVCore::list_elts& elts = out->elements();	
	PVCore::list_elts::iterator it_elt;

	//std::list<QString, tbb::tbb_allocator<QString> > sl;
	/*for (it_elt = elts.begin(); it_elt != elts.end(); it_elt++) {
		PVCore::PVElement& e = *(*it_elt);
		if (!e.valid())
			continue;
		PVCore::list_fields const& fields = e.c_fields();
		if (fields.size() == 0)
			continue;

		if (!_nraw_dest.add_row(e, out)) {
			// Discard the chunk
			out->free();
			return;
		}
	}*/
	
	// Save the chunk corresponding index
	_pvrow_chunk_idx[_nraw_cur_index] = out->agg_index();

	_nraw_cur_index++;

	_nraw_dest.push_chunk(out);

	// Give the ownership of reallocated buffers to the NRAW
	out->give_ownerhsip_realloc_buffers(_nraw_dest);
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
	// Tell the destination NRAW to be created according to the
	// chunk that has been pushed.
	_nraw_dest.fit_to_content();
}
