/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
 */

#ifndef PVSPLUNKSOURCE_FILE_H
#define PVSPLUNKSOURCE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>

#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <boost/bind.hpp>

#include <QString>

#include "../../common/splunk/PVSplunkAPI.h"

namespace PVRush
{

class PVSplunkQuery;

class PVSplunkSource: public PVRawSourceBase
{
public:
	PVSplunkSource(PVInputDescription_p input, PVFilter::PVChunkFilter_f src_filter);

	virtual ~PVSplunkSource();

public:
	QString human_name() override;
	void seek_begin() override;
	bool seek(input_offset /*off*/) override;
	void prepare_for_nelts(chunk_index nelts) override;
	PVCore::PVChunk* operator()() override;

	input_offset get_input_offset_from_index(chunk_index /*idx*/, chunk_index& /*known_idx*/) override
	{
		return 0;
	}

public:
	func_type f() override
	{
		return boost::bind<PVCore::PVChunk*>(&PVSplunkSource::operator(), this);
	}

protected:
	chunk_index _next_index;

private:
	PVSplunkQuery& _query;
	PVSplunkAPI _splunk;
	bool _query_end = false;
};

} // namespace PVRush

#endif
