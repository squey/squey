/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVOutput.h>
#include <pvkernel/core/PVChunk.h>

void PVRush::PVOutput::operator()(PVCore::PVChunk* chunk)
{
	PVLOG_WARN("(PVRush::PVOutput) default output function used !\n");
	chunk->free();
}

void PVRush::PVOutput::set_stop_condition(bool *cond)
{
	_stop_cond = cond;
}
