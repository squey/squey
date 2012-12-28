#ifndef PVARCSIGHTSOURCE_FILE_H
#define PVARCSIGHTSOURCE_FILE_H

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

#include <arcsight/ArcsightLogger.h>
#include <arcsight/ArcsightLoggerSearch.h>

#include "../../common/arcsight/PVArcsightQuery.h"

namespace PVRush {

class PVArcsightSource: public PVRawSourceBase {
public:
	PVArcsightSource(PVInputDescription_p input, size_t min_chunk_size, PVFilter::PVChunkFilter_f src_filter);
	virtual ~PVArcsightSource();
public:
	virtual QString human_name();
	virtual void seek_begin();
	virtual bool seek(input_offset /*off*/) { return false; }
	virtual void prepare_for_nelts(chunk_index nelts);
	virtual PVCore::PVChunk* operator()();
	virtual input_offset get_input_offset_from_index(chunk_index /*idx*/, chunk_index& /*known_idx*/) {return 0;}
public:
	virtual func_type f() { return boost::bind<PVCore::PVChunk*>(&PVArcsightSource::operator(), this); }

protected:
	QString _python_file;
	input_offset _start;
	size_t _min_chunk_size;
	chunk_index _next_index;

private:
	arcsight::ArcsightLogger _arcsight;
	arcsight::ArcsightLoggerSearch _arcsight_search;

	PVArcsightQuery _query; // copy of the current query
};

class PVArcsightInputException: public PVInputException
{
public:
	PVArcsightInputException(const char* msg):
		_msg(msg)
	{ }
public:
	virtual std::string const& what() const override { return _msg; }
private:
	std::string _msg;
};

}

#endif
