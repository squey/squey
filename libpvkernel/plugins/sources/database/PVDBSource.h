/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDBSOURCE_FILE_H
#define PVDBSOURCE_FILE_H

#include <pvbase/general.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include "../../common/database/PVDBQuery.h"

#include <pvkernel/core/PVChunk.h>

#include <QSqlDatabase>

namespace PVRush
{

class PVDBSource : public PVRawSourceBase
{
  public:
	PVDBSource(PVDBQuery const& query, chunk_index nelts_chunk);
	virtual ~PVDBSource();

  public:
	virtual QString human_name();
	virtual void seek_begin();
	bool seek(input_offset off);
	virtual void prepare_for_nelts(chunk_index nelts);
	size_t get_size() const override { return 0; }
	virtual PVCore::PVChunk* operator()();

  protected:
	PVDBQuery _query;
	input_offset _start;
	chunk_index _min_nelts;
	chunk_index _nelts_chunk;
	chunk_index _next_index;
	QSqlQuery _sql_query;
};
}

#endif
