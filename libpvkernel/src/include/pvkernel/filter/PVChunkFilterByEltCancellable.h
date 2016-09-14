/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELTCANCELLABLE_H
#define PVFILTER_PVCHUNKFILTERBYELTCANCELLABLE_H

#include <pvkernel/filter/PVChunkFilter.h>   // for PVChunkFilter
#include <pvkernel/filter/PVElementFilter.h> // for PVElementFilter

#include <memory> // for unique_ptr

namespace PVCore
{
class PVChunk;
} // namespace PVCore

namespace PVFilter
{

class PVChunkFilterByEltCancellable : public PVChunkFilter
{
  public:
	PVChunkFilterByEltCancellable(std::unique_ptr<PVElementFilter> elt_filter,
	                              float timeout,
	                              bool* cancellation = nullptr);
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk) const;

  private:
	std::unique_ptr<PVElementFilter> _elt_filter;

	float _timeout;
	bool* _cancellation;
};
} // namespace PVFilter

#endif
