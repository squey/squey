/**
 * \file PVDBSource.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVDBSOURCE_FILE_H
#define PVDBSOURCE_FILE_H

#include <pvbase/general.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include "../../common/database/PVDBQuery.h"

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <boost/bind.hpp>

#include <QSqlDatabase>

namespace PVRush {

class PVDBSource: public PVRawSourceBase {
public:
	PVDBSource(PVDBQuery const& query, chunk_index nelts_chunk, PVFilter::PVChunkFilter_f src_filter);
	virtual ~PVDBSource();
public:
	virtual QString human_name();
	virtual void seek_begin();
	virtual bool seek(input_offset off);
	virtual void prepare_for_nelts(chunk_index nelts);
	virtual PVCore::PVChunk* operator()();
	virtual input_offset get_input_offset_from_index(chunk_index idx, chunk_index& known_idx);
public:
	virtual func_type f() { return boost::bind<PVCore::PVChunk*>(&PVDBSource::operator(), this); }
private:
	void query_next_batch();
protected:
	PVDBQuery _query;
	input_offset _start;
	chunk_index _min_nelts;
	chunk_index _nelts_chunk;
	chunk_index _next_index;
	QSqlQuery _sql_query;
private:
	//typedef std::vector< PVCore::PVField, tbb::cache_aligned_allocator<PVCore::PVField> > vec_fields_t;
	//vec_fields_t _vec_fields;
};

}

#endif
