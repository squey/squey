/**
 * \file PVOutput.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVOutput.h>
#include <pvkernel/core/PVChunk.h>

void PVRush::PVOutput::operator()(PVCore::PVChunk* chunk)
{
	PVLOG_WARN("(PVRush::PVOutput) default output function used !\n");
	chunk->free();
}
