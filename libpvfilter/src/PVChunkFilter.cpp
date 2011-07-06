//! \file PVChunkFilter.cpp
//! $Id: PVChunkFilter.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvfilter/PVChunkFilter.h>
#include <pvcore/PVChunk.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilter::PVChunkFilter
 *
 *****************************************************************************/
PVFilter::PVChunkFilter::PVChunkFilter() :
	PVFilterFunctionBase<PVCore::PVChunk*,PVCore::PVChunk*>()
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilter::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilter::operator()(PVCore::PVChunk *chunk)
{
	return chunk;
}

