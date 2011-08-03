#include <pvkernel/rush/PVOutput.h>
#include <pvkernel/core/PVChunk.h>

void PVRush::PVOutput::operator()(PVCore::PVChunk* chunk)
{
	chunk->free();
}
