/**
 * \file PVPerlSource.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPerlSource.h"
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

PVRush::PVPerlSource::PVPerlSource(input_type input, size_t min_chunk_size, PVFilter::PVChunkFilter_f src_filter, const QString& perl_file):
	PVRawSourceBase(src_filter),
	_perl_file(perl_file),
	_min_chunk_size(min_chunk_size),
	_next_index(0)
{
	SV* svp;

	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	QByteArray file_str(file->path().toLocal8Bit());

	my_perl = perl_alloc();
	perl_construct(my_perl);

	QByteArray perl_file_str(perl_file.toLocal8Bit());

	char *args[2] = {(char*) "", perl_file_str.data()} ;

	if (perl_parse(my_perl, NULL, 2, args, (char **)NULL) != 0) {
		throw PVPerlFormatInvalid(perl_file);
	}

	dSP;
	ENTER;
	SAVETMPS;
	PUSHMARK(SP);
	XPUSHs(sv_2mortal(newSVpv(file_str.data(), 0)));
	PUTBACK;
	perl_call_pv("picviz_open_file", G_ARRAY | G_EVAL);
	SPAGAIN;
	svp = GvSV(gv_fetchpv("@", TRUE, SVt_PV));
	if (SvTRUE(svp)) {
		char* err_msg = SvPV_nolen(svp);
		POPs;
		throw PVPerlExecException(perl_file, err_msg);
	}

	PUTBACK;
	FREETMPS;
	LEAVE; 
}

PVRush::PVPerlSource::~PVPerlSource()
{
	SV* svp;
	dSP;
	ENTER;
	SAVETMPS;
	PUSHMARK(SP);
	PUTBACK;
	perl_call_pv("picviz_close", G_EVAL);
	SPAGAIN;
	svp = GvSV(gv_fetchpv("@", TRUE, SVt_PV));
	if (SvTRUE(svp)) {
		char* err_msg = SvPV_nolen(svp);
		POPs;
		PVLOG_ERROR("Error in Perl file %s: %s\n", qPrintable(_perl_file), err_msg);
		return;
	}

	PUTBACK;
	FREETMPS;
	LEAVE; 

	perl_destruct(my_perl);
	perl_free(my_perl);
}

QString PVRush::PVPerlSource::human_name()
{
	return QString("perl:toto.pl");
}

void PVRush::PVPerlSource::seek_begin()
{
	// AG: if I understand correctly, we should be using something
	// like this:
	// perl_call_pv("picviz_seek_begin", G_DISCARD | G_NOARGS);
	// because w have no argument (so no need for Perl stack stuff) and
	// we discard the output of the function.
	// Still, it does not work (segfault) so we are using the code below...
	
	SV* svp;
	dSP;
	ENTER;
	SAVETMPS;
	PUSHMARK(SP);
	PUTBACK;
	perl_call_pv("picviz_seek_begin", G_EVAL);
	SPAGAIN;
	svp = GvSV(gv_fetchpv("@", TRUE, SVt_PV));
	if (SvTRUE(svp)) {
		char* err_msg = SvPV_nolen(svp);
		POPs;
		PVLOG_ERROR("Error in Perl file %s: %s\n", qPrintable(_perl_file), err_msg);
		return;
	}

	PUTBACK;
	FREETMPS;
	LEAVE; 
}

void PVRush::PVPerlSource::prepare_for_nelts(chunk_index /*nelts*/)
{
}

PVCore::PVChunk* PVRush::PVPerlSource::operator()()
{
	SV* svp;
	AV *av;
	SV *sv;
	SV *array_sv;
	SV **array_elem_pp;
	SV *array_elem;
	int nelts;

	// Initialize PERL stack
	dSP;
	ENTER;
	SAVETMPS;
	PUSHMARK(SP);
	mXPUSHu((UV) _min_chunk_size);
	PUTBACK;
	nelts = perl_call_pv("picviz_get_next_chunk", G_ARRAY | G_EVAL);
	SPAGAIN;
	svp = GvSV(gv_fetchpv("@", TRUE, SVt_PV));
	if (SvTRUE(svp)) {
		char* err_msg = SvPV_nolen(svp);
		POPs;
		PVLOG_ERROR("Error in Perl file %s: %s\n", qPrintable(_perl_file), err_msg);
		return NULL;
	}

	if (nelts == 0) {
		// That's the end
		return NULL;
	}

	PVCore::PVChunk* chunk = PVCore::PVChunkMem<>::allocate(0, this);
	chunk->set_index(_next_index);

	for (int i = 0; i < nelts; i++) {
		sv = POPs;
		array_sv = SvRV(sv);
		av = (AV *)array_sv;

		PVCore::PVElement* elt = chunk->add_element();
		elt->fields().clear();
		ssize_t len_ar = av_len(av);
		if (len_ar < 0) {
			continue;
		}
		for (int j = 0; j <= av_len(av); j++) {
			array_elem_pp = av_fetch(av, j, FALSE);
			array_elem = *array_elem_pp;

			char* field_buf;
			if (SvPOK(array_elem)) {
				field_buf = SvPV_nolen(array_elem);
			}
			else {
				PVLOG_ERROR("Field %d of chunk element %d does not contain a string. Using an empty one...\n", i, j);
				field_buf = (char*) "";
			}
			QString value = QString::fromUtf8(field_buf);
			PVCore::PVField f(*elt);
			size_t size_buf = value.size() * sizeof(QChar);
			f.allocate_new(size_buf);
			memcpy(f.begin(), value.constData(), size_buf);
			elt->fields().push_back(f);
		}
	}

	// Clear PERL stack
	PUTBACK;
	FREETMPS;
	LEAVE; 

	// Set the index of the elements inside the chunk
	chunk->set_elements_index();

	// Compute the next chunk's index
	_next_index += chunk->c_elements().size();
	if (_next_index-1>_last_elt_index) {
		_last_elt_index = _next_index-1;
	}

	return chunk;
}

