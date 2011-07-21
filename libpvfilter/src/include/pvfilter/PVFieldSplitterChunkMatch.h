#ifndef PVFIELDSPLITTERCHUNKMATCH_H
#define PVFIELDSPLITTERCHUNKMATCH_H

#include <pvcore/general.h>
#include <pvfilter/PVFieldsFilter.h>
#include <pvfilter/PVRawSourceBase.h>

namespace PVFilter {

class LibFilterDecl PVFieldSplitterChunkMatch
{
public:
	PVFieldSplitterChunkMatch(PVFilter::PVFieldsSplitter_p filter) :
		_filter(filter)
	{
	}

	void push_chunk(PVCore::PVChunk* chunk);
	bool get_match(PVCore::PVArgumentList& args, size_t& nfields);

	static PVFilter::PVFieldsSplitter_p get_match_on_input(PVFilter::PVRawSourceBase_p src);

protected:
	PVFilter::list_guess_result_t _guess_res;
	PVFilter::PVFieldsSplitter_p _filter;
};

}

#endif
