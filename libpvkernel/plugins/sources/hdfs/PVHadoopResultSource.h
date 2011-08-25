#ifndef PVHADOOPRESULTSOURCE_FILE_H
#define PVHADOOPRESULTSOURCE_FILE_H

#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include "PVHadoopResultServer.h"

namespace PVRush {

class PVHadoopResultSource: public PVRawSourceBase
{
public:
	PVHadoopResultSource(PVInputHDFSFile const& file, PVCol nfields, size_t chunk_size):
		PVRawSourceBase(_null_filter.f()),
		_recv_serv(nfields, chunk_size)
	{
		_name = file.get_human_name();
		_recv_serv.start(1245);
	}

	~PVHadoopResultSource()
	{
		_recv_serv.stop();
	}

public:
	QString human_name() { return _name; }
	void seek_begin() { }
	bool seek(input_offset /*off*/) { return false; }
	void prepare_for_nelts(chunk_index /*nelts*/) { }
	PVCore::PVChunk* operator()()
	{
		PVCore::PVChunk* ret = _recv_serv();
		if (ret == NULL) {
			return NULL;
		}

		_last_elt_index = ret->last_elt_index();
		return ret;
	}
	input_offset get_input_offset_from_index(chunk_index /*idx*/, chunk_index& /*known_idx*/) { return 0; }

	virtual func_type f() { return boost::bind<PVCore::PVChunk*>(&PVHadoopResultSource::operator(), this); }
protected:
	PVHadoopResultServer _recv_serv;
	QString _name;
	PVFilter::PVChunkFilter _null_filter;
};

}

#endif
