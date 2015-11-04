/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <boost/function.hpp>

#include <pvkernel/filter/PVChunkFilterSource.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/core/PVChunk.h>


/******************************************************************************
 *
 * PVFilter::PVChunkFilterSource::PVChunkFilterSource
 *
 *****************************************************************************/
PVFilter::PVChunkFilterSource::PVChunkFilterSource() :
	PVChunkFilter()
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterSource::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterSource::operator()(PVCore::PVChunk* chunk)
{
	return (chunk->source()->source_filter())(chunk);
}

