/**
 * \file PVChunkFilterSource.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
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

