#ifndef PVINPUTHADOOP_FILE_H
#define PVINPUTHADOOP_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVInput.h>

#include <boost/shared_ptr.hpp>

#include "../../common/hdfs/PVInputHDFSFile.h"
#include "PVChunkAlignHadoop.h"
#include "PVHadoopResultServer.h"

namespace PVRush {

class PVInputHadoop {
	friend class PVChunkAlignHadoop;
public:
	PVInputHadoop(PVInputHDFSFile const& file);
	~PVInputHadoop();
public:
	virtual size_t operator()(char* buffer, size_t n);
	virtual input_offset current_input_offset();
	virtual void seek_begin();
	virtual QString human_name();
	virtual bool seek(input_offset off);
	PVChunkAlignHadoop& get_align() { return _align; }

protected:
	inline void set_last_seen_offset(input_offset off) { _last_off = off; }

protected:
	PVChunkAlignHadoop _align;
	input_offset _last_off;
	PVHadoopResultServer _recv_serv;
	QString _name;

	CLASS_INPUT(T)
};

}

#endif
