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
	void set_should_process_in_hadoop(bool b) { _process_in_hadoop = b; }
	bool should_process_in_hadoop(bool b) const { return _process_in_hadoop; }
protected:
	PVInputHDFSFile _file_param;
	hdfsFS _fs;
	hdfsFile _file;
	bool _process_in_hadoop;

	CLASS_INPUT(PVRush::PVInputHDFS)
};

}

#endif
