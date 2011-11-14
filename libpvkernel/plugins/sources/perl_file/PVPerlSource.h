#ifndef PVPERLSOURCE_FILE_H
#define PVPERLSOURCE_FILE_H

#include <QString>

#include <pvbase/general.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <boost/bind.hpp>

#include <EXTERN.h>
#include <perl.h>

namespace PVRush {

class PVPerlSource: public PVRawSourceBase {
public:
	PVPerlSource(input_type input, size_t min_chunk_size, PVFilter::PVChunkFilter_f src_filter, const QString& perl_file);
	virtual ~PVPerlSource();
public:
	virtual QString human_name();
	virtual void seek_begin();
	virtual bool seek(input_offset /*off*/) { return false; }
	virtual void prepare_for_nelts(chunk_index nelts);
	virtual PVCore::PVChunk* operator()();
	virtual input_offset get_input_offset_from_index(chunk_index /*idx*/, chunk_index& /*known_idx*/) {return 0;}
public:
	virtual func_type f() { return boost::bind<PVCore::PVChunk*>(&PVPerlSource::operator(), this); }
protected:
	QString _perl_file;
	input_offset _start;
	size_t _min_chunk_size;
	chunk_index _next_index;
private:
	static PerlInterpreter *my_perl;
};

}

#endif
