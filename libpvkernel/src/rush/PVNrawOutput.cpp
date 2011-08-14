#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>

#include <tbb/tbb_allocator.h>

PVRush::PVNrawOutput::PVNrawOutput(PVRush::PVNraw &nraw_dest) :
	_nraw_dest(nraw_dest)
{
}

void PVRush::PVNrawOutput::operator()(PVCore::PVChunk* out)
{
	// Write all elements of the chunk in the final nraw
	PVCore::list_elts const& elts = out->c_elements();	
	PVCore::list_elts::const_iterator it_elt;

	// Index used for the pvrow map table
	PVRow nraw_index = _nraw_dest.table.size();

	//std::list<QString, tbb::tbb_allocator<QString> > sl;
	for (it_elt = elts.begin(); it_elt != elts.end(); it_elt++) {
		PVCore::PVElement const& e = *it_elt;
		if (!e.valid())
			continue;
		PVCore::list_fields const& fields = e.c_fields();
		PVCore::list_fields::const_iterator it_field;
		if (fields.size() == 0)
			continue;

		//PVLOG_DEBUG("(PVNrawOutput) add element\n");
		
		size_t nchars_line = 0;
		PVRush::PVNraw::nraw_table_line &sl = _nraw_dest.add_row(fields.size());
		size_t index_f = 0;
		for (it_field = fields.begin(); it_field != fields.end(); it_field++) {
			PVCore::PVField const& f = *it_field;
			if (!f.valid())
				continue;
			//PVLOG_DEBUG("(PVNrawOutput) add field\n");
			nchars_line += f.size();
			//sl[index_f].setUnicode((QChar*) f.begin(), f.size()/(sizeof(QChar)));
			_nraw_dest.set_field(sl, index_f, (QChar*) f.begin(), f.size()/(sizeof(QChar)));
			index_f++;
		}
		_nraw_dest.push_line_chars(nchars_line);
	}
	
	// Save the chunk corresponding index
	_pvrow_chunk_idx[nraw_index] = out->agg_index();
	
	// Free the chunk
	out->free();
}

PVRush::PVNrawOutput::map_pvrow const& PVRush::PVNrawOutput::get_pvrow_index_map() const
{
	return _pvrow_chunk_idx;
}

void PVRush::PVNrawOutput::clear_pvrow_index_map()
{
	_pvrow_chunk_idx.clear();
}

void PVRush::PVNrawOutput::job_has_finished()
{
	// Tell the destination NRAW to resize its content
	// to what it actually has, in case too much
	// elements have been pre-allocated.
	_nraw_dest.fit_to_content();
}
