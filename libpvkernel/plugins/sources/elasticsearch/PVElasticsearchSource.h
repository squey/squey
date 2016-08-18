/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#ifndef PVELASTICSEARCHSOURCE_FILE_H
#define PVELASTICSEARCHSOURCE_FILE_H

#include <QString>

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include "../../common/elasticsearch/PVElasticsearchAPI.h"

namespace PVRush
{

class PVElasticsearchSource : public PVRawSourceBase
{
  public:
	PVElasticsearchSource(PVInputDescription_p input);
	virtual ~PVElasticsearchSource();

  public:
	QString human_name() override;
	void seek_begin() override;
	void prepare_for_nelts(chunk_index nelts) override;
	PVCore::PVChunk* operator()() override;

  protected:
	chunk_index _next_index;

  private:
	PVElasticsearchQuery& _query;
	PVElasticsearchAPI _elasticsearch;
	bool _query_end = false;
	std::string _scroll_id;
};
}

#endif
