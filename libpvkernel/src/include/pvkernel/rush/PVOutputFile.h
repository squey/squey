/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVOUTPUTFILE_FILE_H
#define PVOUTPUTFILE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVOutput.h>
#include <stdio.h>

namespace PVRush {

class PVOutputFile : public PVOutput {
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
