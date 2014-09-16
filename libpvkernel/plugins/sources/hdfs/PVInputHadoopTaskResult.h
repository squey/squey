/**
 * \file PVInputHadoopTaskResult.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINPUTHADOOPTASKRESULT_FILE_H
#define PVINPUTHADOOPTASKRESULT_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVInput.h>

#include <memory>

#include "PVChunkAlignHadoop.h"
#include "PVHadoopTaskResult.h"

namespace PVRush {

class PVInputHadoopTaskResult: public PVInput {
	friend class PVChunkAlignHadoop;
public:
	typedef std::shared_ptr<PVInputHadoopTaskResult> p_type;
public:
	PVInputHadoopTaskResult(PVHadoopTaskResult_p task, PVCol nfields):
		_align(*this,nfields), _last_off(0), _task(task)
	{
		_last_off = 0;
	}
	~PVInputHadoopTaskResult() {} ;
public:
	virtual size_t operator()(char* buffer, size_t n) { return _task->read_sock(buffer, n); }
	virtual input_offset current_input_offset() { return _last_off; }
	virtual void seek_begin() { }
	virtual QString human_name() { return QString("TODO: name of task"); };
	virtual bool seek(input_offset /*off*/) { return false; }
	PVChunkAlignHadoop& get_align() { return _align; }
	PVHadoopTaskResult_p task() { return _task; }

protected:
	inline void set_last_seen_offset(input_offset off) { _last_off = off; }

protected:
	PVChunkAlignHadoop _align;
	input_offset _last_off;
	PVHadoopTaskResult_p _task;

	CLASS_INPUT(PVRush::PVInputHadoopTaskResult)
};

typedef PVInputHadoopTaskResult::p_type PVInputHadoopTaskResult_p;

}

#endif
