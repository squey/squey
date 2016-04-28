/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/core/PVChunk.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilter::PVChunkFilter
 *
 *****************************************************************************/
PVFilter::PVChunkFilter::PVChunkFilter()
    : PVFilterFunctionBase<PVCore::PVChunk*, PVCore::PVChunk*>()
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilter::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilter::operator()(PVCore::PVChunk* chunk)
{
	return chunk;
}
