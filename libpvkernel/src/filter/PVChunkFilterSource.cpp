//! \file PVChunkFilterSource.cpp
//! $Id: PVChunkFilterSource.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <boost/function.hpp>

#include <pvkernel/filter/PVChunkFilterSource.h>
#include <pvkernel/filter/PVRawSourceBase.h>
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

