/**
 * \file PVOutput.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVOUTPUT_FILE_H
#define PVOUTPUT_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>

namespace PVRush {

class PVControllerJob;

class LibKernelDecl PVOutput : public PVFilter::PVFilterFunctionBase<void,PVCore::PVChunk*> {
	friend class PVControllerJob;
public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	void operator()(PVCore::PVChunk* out);

public:
	virtual PVRow get_rows_count() = 0;

protected:
	// This function is called by PVControllerJob
	// when its job has finished.
	virtual void job_has_finished() {}

	CLASS_FILTER_NONREG(PVOutput)
};

}

#endif
