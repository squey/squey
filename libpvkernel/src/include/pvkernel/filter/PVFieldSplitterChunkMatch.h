/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELDSPLITTERCHUNKMATCH_H
#define PVFIELDSPLITTERCHUNKMATCH_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>

namespace PVFilter
{

class PVFieldSplitterChunkMatch
{
  public:
	PVFieldSplitterChunkMatch(PVFilter::PVFieldsSplitter_p filter) : _filter(filter) {}

	void push_chunk(PVCore::PVChunk* chunk);
	bool get_match(PVCore::PVArgumentList& args, size_t& nfields);

	static PVFilter::PVFieldsSplitter_p get_match_on_input(PVRush::PVRawSourceBase_p src,
	                                                       PVCol& naxes);

  protected:
	PVFilter::list_guess_result_t _guess_res;
	PVFilter::PVFieldsSplitter_p _filter;
};
}

#endif
