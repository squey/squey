/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#ifndef PVELASTICSEARCHSOURCE_FILE_H
#define PVELASTICSEARCHSOURCE_FILE_H

#include <QString>

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include "../../common/elasticsearch/PVElasticsearchAPI.h"

namespace PVRush
{

class PVElasticsearchSource : public PVRawSourceBaseType<PVCore::PVTextChunk>
{
  public:
	PVElasticsearchSource(PVInputDescription_p input);
	virtual ~PVElasticsearchSource();

  public:
	QString human_name() override;
	void seek_begin() override;
	void prepare_for_nelts(chunk_index nelts) override;

	/**
	 * Return the total size of this source.
	 *
	 * It use line count for ES.
	 */
	size_t get_size() const override;
	PVCore::PVTextChunk* operator()() override;

  protected:
	chunk_index _next_index;

  private:
	PVElasticsearchQuery& _query;
	PVElasticsearchAPI _elasticsearch;
	bool _query_end = false;
	std::string _scroll_id;
};
} // namespace PVRush

#endif
