/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDBSOURCE_FILE_H
#define PVDBSOURCE_FILE_H

#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInput_types.h>

#include <pvkernel/core/PVTextChunk.h>

#include "../../common/database/PVDBQuery.h"

#include <QSqlDatabase>

namespace PVRush
{

class PVDBSource : public PVRawSourceBaseType<PVCore::PVTextChunk>
{
  public:
	PVDBSource(PVDBQuery const& query, chunk_index nelts_chunk);
	virtual ~PVDBSource();

  public:
	QString human_name() override;
	void seek_begin() override;
	void prepare_for_nelts(chunk_index nelts) override;
	size_t get_size() const override { return 0; }
	PVCore::PVTextChunk* operator()() override;

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
