/**
 * \file PVInputHDFS.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINPUTHDFS_FILE_H
#define PVINPUTHDFS_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInput.h>

#include <hadoop/hdfs.h>

#include <QString>

#include "../../common/hdfs/PVInputHDFSFile.h"

namespace PVRush {


class PVInputHDFS : public PVInput {
public:
	PVInputHDFS(PVInputHDFSFile const& in);
public:
	size_t operator()(char* buffer, size_t n);
	virtual input_offset current_input_offset();
	virtual void seek_begin();
	virtual bool seek(input_offset off);
	virtual QString human_name();
protected:
	PVInputHDFSFile _file_param;
	hdfsFS _fs;
	hdfsFile _file;

	CLASS_INPUT(PVRush::PVInputHDFS)
};

}

#endif
