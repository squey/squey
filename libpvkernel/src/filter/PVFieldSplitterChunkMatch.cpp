#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>
#include <pvkernel/filter/PVFilterLibrary.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <list>
#include <utility>


void PVFilter::PVFieldSplitterChunkMatch::push_chunk(PVCore::PVChunk* chunk)
{
	PVFilter::PVFieldsSplitter_p sp = _filter;
	PVCore::list_fields lf_res;
	PVCore::PVArgumentList args_match;

	PVCore::list_elts const& le = chunk->c_elements();
	PVCore::list_elts::const_iterator it_elt;
	for (it_elt = le.begin(); it_elt != le.end(); it_elt++) {
		PVCore::PVField const& first_f = it_elt->c_fields().front();
		sp->guess(_guess_res, first_f);
	}
}

bool PVFilter::PVFieldSplitterChunkMatch::get_match(PVCore::PVArgumentList& args, size_t& nfields)
{
	// Reduce _guess_res and check which args always returns the same number of fields
	typedef std::list< std::pair<PVCore::PVArgumentList, size_t> > list_args_n_t;
	list_args_n_t red;

	PVFilter::list_guess_result_t::iterator it;
	for (it = _guess_res.begin(); it != _guess_res.end(); it++) {
		PVCore::PVArgumentList const& args_test = it->first;
		size_t nfields = it->second.size();

		// Look for "args" in "red"
		list_args_n_t::iterator it_red_args;
		for (it_red_args = red.begin(); it_red_args != red.end(); it_red_args++) {
			if (it_red_args->first == args_test) {
				break;
			}
		}

		if (it_red_args != red.end() && it_red_args->second != nfields) {
			// This is an invalid set of arguments. Clear it for now.
			red.erase(it_red_args);
		}
		else {
			// This is a new set of args. Add it to the "reduction" list.
			red.push_back(list_args_n_t::value_type(args_test, nfields));
		}
	}

	if (red.size() == 0) {
		// No match here
		return false;
	}

	args = red.front().first;
	nfields = red.front().second;

	return true;
}

PVFilter::PVFieldsSplitter_p PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(PVRush::PVRawSourceBase_p src, PVCol &naxes)
{
	PVCore::PVChunk* chunk = (*src)();
	PVFieldsSplitter_p ret;
	if (!chunk) {
		src->seek_begin();
		chunk = (*src)();
		if (!chunk) {
			return ret;
		}
	}
	LIB_FILTER(PVFilter::PVFieldsSplitter)::list_filters const& lf = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_list();
	LIB_FILTER(PVFilter::PVFieldsSplitter)::list_filters::const_iterator it;
	for (it = lf.begin(); it != lf.end(); it++) {
		PVFilter::PVFieldsSplitter_p sp = (*it)->clone<PVFilter::PVFieldsSplitter>();
		PVFilter::PVFieldSplitterChunkMatch match(sp);
		match.push_chunk(chunk);

		PVCore::PVArgumentList args;
		size_t nfields;

		if (match.get_match(args, nfields)) {
			PVLOG_DEBUG("(PVFieldSplitterChunkMatch) filter %s matches with %d fields\n with arguments:\n", qPrintable(it.key()), nfields);
			PVCore::dump_argument_list(args);
			ret = sp;
			ret->set_number_expected_fields(nfields);
			ret->set_args(args);
			naxes = nfields;
			break;
		}
	}
	chunk->free();

	return ret;
}
