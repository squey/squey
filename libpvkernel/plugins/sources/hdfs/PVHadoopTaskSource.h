/**
 * \file PVHadoopTaskSource.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVHADOOPTASKSOURCE_FILE_H
#define PVHADOOPTASKSOURCE_FILE_H

#include "PVInputHadoopTaskResult.h"
#include "PVChunkTransformHadoop.h"
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/rush/PVRawSource.h>

#include <queue>

#include <boost/thread/thread.hpp>

#include <tbb/scalable_allocator.h>
#include <tbb/concurrent_queue.h>

namespace PVRush {

class PVHadoopTaskSource {
public:
	typedef boost::shared_ptr<PVHadoopTaskSource> p_type;
	typedef tbb::concurrent_queue<PVCore::PVChunk*> list_pchunks;

private:
	PVHadoopTaskSource(PVInputHadoopTaskResult_p input_task, size_t chunk_size):
		_input(input_task)
	{
		_read_finished = false;
		_source.reset(new PVRawSource<>(PVInput_p(_input), (PVChunkAlign&) _input->get_align(), chunk_size, (PVChunkTransform&) _trans, _null_filter.f()));

		// Create a thread that create the chunks as they come from this task.
		_should_stop = false;
		_thread_read = boost::thread(boost::bind(&PVHadoopTaskSource::read_chunks, this));
	}

public:
	virtual ~PVHadoopTaskSource()
	{
		_should_stop = true;
		_thread_read.join();
	}

public:
	static p_type create_from_task(PVHadoopTaskResult_p task, PVCol nfields, size_t chunk_size)
	{
		return p_type(new PVHadoopTaskSource(PVInputHadoopTaskResult_p(new PVInputHadoopTaskResult(task, nfields)), chunk_size));
	}

	PVCore::PVChunk* operator()()
	{
		PVCore::PVChunk* ret;
		while (!_chunks.try_pop(ret)) {
			// If no more chunks will be read, returns NULL as this is
			// the end for this task.
			if (_read_finished) {
				return NULL;
			}
			// Else, wait for a new chunk to be read...
			{
				boost::unique_lock<boost::mutex> lock(_chunk_added_mutex);
				while (_chunks.empty()) {
					_chunk_added.wait(lock);
				}
			}
		}

		return ret;
	}

	bool job_finished() { return _input->task()->job_finished(); }
	PVHadoopTaskResult::id_type id() { return _input->task()->id(); }

private:
	void read_chunks()
	{
		_read_finished = false;
		PVCore::PVChunk* chunk;
		while (!_should_stop && ((chunk = _source->operator()()) != NULL)) {
			_chunks.push(chunk);
			_chunk_added.notify_all();
		}
		_read_finished = true;
	}

public:
	PVRawSourceBase::p_type _source;
	PVInputHadoopTaskResult_p _input;
	PVChunkTransformHadoop _trans;
	PVFilter::PVChunkFilter _null_filter;
	list_pchunks _chunks;
	bool _read_finished;

	boost::condition_variable _chunk_added;
	boost::mutex _chunk_added_mutex;

	boost::thread _thread_read;
	bool _should_stop;
};

typedef PVHadoopTaskSource::p_type PVHadoopTaskSource_p;


}

#endif
