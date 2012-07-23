/**
 * \file PVChunkFilter.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/core/PVChunk.h>

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

