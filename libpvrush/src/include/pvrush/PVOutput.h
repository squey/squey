#ifndef PVOUTPUT_FILE_H
#define PVOUTPUT_FILE_H

#include <pvcore/general.h>
#include <pvcore/PVChunk.h>
#include <pvfilter/PVFilterFunction.h>

namespace PVRush {

class LibRushDecl PVOutput : public PVFilter::PVFilterFunctionBase<void,PVCore::PVChunk*> {
public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	virtual void operator()(PVCore::PVChunk* out);
};

}

#endif
