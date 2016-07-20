/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELTCANCELLABLE_H
#define PVFILTER_PVCHUNKFILTERBYELTCANCELLABLE_H

#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter
{

class PVChunkFilterByEltCancellable : public PVChunkFilter
{
  public:
	PVChunkFilterByEltCancellable(PVElementFilter_f elt_filter,
	                              float timeout,
	                              bool* cancellation = nullptr);
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk) const;

  private:
	PVElementFilter_f _elt_filter;

	float _timeout;
	bool* _cancellation;
};
}

#endif
