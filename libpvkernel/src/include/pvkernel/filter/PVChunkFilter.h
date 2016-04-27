/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVCHUNKFILTER_H
#define PVFILTER_PVCHUNKFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>

namespace PVFilter
{

class PVChunkFilter : public PVFilterFunctionBase<PVCore::PVChunk*, PVCore::PVChunk*>
{
  public:
	PVChunkFilter();

  public:
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);

	CLASS_FILTER_NONREG(PVChunkFilter)
};

typedef PVChunkFilter::func_type PVChunkFilter_f;
}

#endif
