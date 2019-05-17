/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#ifndef PVSPLUNKSOURCE_FILE_H
#define PVSPLUNKSOURCE_FILE_H

#include <pvkernel/core/PVTextChunk.h>

#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <QString>

#include "../../common/splunk/PVSplunkAPI.h"

namespace PVRush
{

class PVSplunkQuery;

class PVSplunkSource : public PVRawSourceBaseType<PVCore::PVTextChunk>
{
  public:
	PVSplunkSource(PVInputDescription_p input);

	virtual ~PVSplunkSource();

  public:
	QString human_name() override;
	void seek_begin() override;
	void prepare_for_nelts(chunk_index nelts) override;

	/**
	 * It should be the size of the Splunk source.
	 *
	 * As we don't have "free" metric, we return 0 for "unknown"
	 */
	size_t get_size() const override { return 0; }
	PVCore::PVTextChunk* operator()() override;

  protected:
	chunk_index _next_index;

  private:
	PVSplunkQuery& _query;
	PVSplunkAPI _splunk;
	bool _query_end = false;
};

} // namespace PVRush

#endif
