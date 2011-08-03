#ifndef PVOUTPUTFILE_FILE_H
#define PVINPUTFILE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVOutput.h>
#include <stdio.h>

namespace PVRush {

class LibKernelDecl PVOutputFile : public PVOutput {
public:
	PVOutputFile(const char* path);
	~PVOutputFile();
public:
	virtual void operator()(PVCore::PVChunk* out);
protected:
	mutable FILE* _file;
};

}


#endif
