#include "PVPerlSource.h"
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

extern "C" {
#include <EXTERN.h>
#include <perl.h>
}

PerlInterpreter *PVRush::PVPerlSource::my_perl = NULL;

PVRush::PVPerlSource::PVPerlSource(input_type input, size_t min_chunk_size, PVFilter::PVChunkFilter_f src_filter, const QString& perl_file):
	PVRawSourceBase(src_filter),
	_perl_file(perl_file)
{
	char *args[2];
	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	QByteArray file_str(file->path().toLocal8Bit());

	if (!my_perl) {
		my_perl = perl_alloc();
		perl_construct(my_perl);
	}

	QByteArray perl_file_str(perl_file.toLocal8Bit());

        args[0] = "";	
	args[1] = perl_file_str.data();

	perl_parse(my_perl, NULL, 2, args, (char **)NULL);
        perl_run(my_perl);

	dSP;
	ENTER;
      	SAVETMPS;
      	PUSHMARK(SP);
        XPUSHs(sv_2mortal(newSVpv(file_str.data(), 0)));
      	PUTBACK;
      	perl_call_pv("picviz_open_file", G_ARRAY);
      	SPAGAIN;

        PUTBACK;
        FREETMPS;
        LEAVE; 

}

PVRush::PVPerlSource::~PVPerlSource()
{
	// perl_destruct(perl_interpreter);
	// perl_free(perl_interpreter);
}

QString PVRush::PVPerlSource::human_name()
{
	return QString("perl:toto.pl");
}

void PVRush::PVPerlSource::seek_begin()
{
}

void PVRush::PVPerlSource::prepare_for_nelts(chunk_index nelts)
{
}

PVCore::PVChunk* PVRush::PVPerlSource::operator()()
{
	return NULL;
}

