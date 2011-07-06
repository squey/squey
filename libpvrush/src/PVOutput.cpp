#include <pvrush/PVOutput.h>
#include <pvcore/PVChunk.h>

void PVRush::PVOutput::operator()(PVCore::PVChunk* chunk)
{
	chunk->free();
}
