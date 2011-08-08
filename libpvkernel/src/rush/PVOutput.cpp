#include <pvkernel/rush/PVOutput.h>
#include <pvkernel/core/PVChunk.h>

void PVRush::PVOutput::operator()(PVCore::PVChunk* chunk)
{
	PVLOG_WARN("(PVRush::PVOutput) default output function used !\n");
	chunk->free();
}
