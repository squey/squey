/**
 * \file PVElasticsearchSource.h
 *
 * Copyright (C) Picviz Labs 2015
 */

#ifndef PVELASTICSEARCHSOURCE_FILE_H
#define PVELASTICSEARCHSOURCE_FILE_H

#include <QString>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <boost/bind.hpp>

#include "../../common/elasticsearch/PVElasticsearchAPI.h"

namespace PVRush {

class PVElasticsearchSource: public PVRawSourceBase {
public:
	PVElasticsearchSource(PVInputDescription_p input, PVFilter::PVChunkFilter_f src_filter);
	virtual ~PVElasticsearchSource();

public:
	QString human_name() override;
	void seek_begin() override;
	bool seek(input_offset /*off*/) override;
	void prepare_for_nelts(chunk_index nelts) override;
	PVCore::PVChunk* operator()() override;
	input_offset get_input_offset_from_index(chunk_index /*idx*/, chunk_index& /*known_idx*/) override {return 0;}

public:
	func_type f() override { return boost::bind<PVCore::PVChunk*>(&PVElasticsearchSource::operator(), this); }

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
