#ifndef PVOUTPUTFILE_FILE_H
#define PVINPUTFILE_FILE_H

#include <pvcore/general.h>
#include <pvrush/PVOutput.h>
#include <stdio.h>

namespace PVRush {

class LibRushDecl PVOutputFile : public PVOutput {
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
