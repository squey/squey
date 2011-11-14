#include "PVPythonSource.h"
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVFileDescription.h>

PythonInterpreter *PVRush::PVPythonSource::my_python = NULL;

PVRush::PVPythonSource::PVPythonSource(input_type input, size_t min_chunk_size, PVFilter::PVChunkFilter_f src_filter, const QString& python_file):
	PVRawSourceBase(src_filter),
	_python_file(python_file),
	_min_chunk_size(min_chunk_size),
	_next_index(0)
{
	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	QByteArray file_str(file->path().toLocal8Bit());

	if (!my_python) {
		my_python = python_alloc();
		python_construct(my_python);
	}

	QByteArray python_file_str(python_file.toLocal8Bit());

	char *args[2] = {(char*) "", python_file_str.data()} ;

	python_parse(my_python, NULL, 2, args, (char **)NULL);

	dSP;
	ENTER;
	SAVETMPS;
	PUSHMARK(SP);
	XPUSHs(sv_2mortal(newSVpv(file_str.data(), 0)));
	PUTBACK;
	python_call_pv("picviz_open_file", G_ARRAY);
	SPAGAIN;

	PUTBACK;
	FREETMPS;
	LEAVE; 
}

PVRush::PVPythonSource::~PVPythonSource()
{
	// python_destruct(python_interpreter);
	// python_free(python_interpreter);
}

QString PVRush::PVPythonSource::human_name()
{
	return QString("python:toto.pl");
}

void PVRush::PVPythonSource::seek_begin()
{
	// AG: if I understand correctly, we should be using something
	// like this:
	// python_call_pv("picviz_seek_begin", G_DISCARD | G_NOARGS);
	// because w have no argument (so no need for Python stack stuff) and
	// we discard the output of the function.
	// Still, it does not work (segfault) so we are using the code below...
	
	dSP;
	ENTER;
	SAVETMPS;
	PUSHMARK(SP);
	PUTBACK;
	python_call_pv("picviz_seek_begin", 0);
	SPAGAIN;

	PUTBACK;
	FREETMPS;
	LEAVE; 
}

void PVRush::PVPythonSource::prepare_for_nelts(chunk_index /*nelts*/)
{
}

PVCore::PVChunk* PVRush::PVPythonSource::operator()()
{
	AV *av;
	SV *sv;
	SV *array_sv;
	SV **array_elem_pp;
	SV *array_elem;
	int nelts;

	// Initialize PYTHON stack
	dSP;
	ENTER;
	SAVETMPS;
	PUSHMARK(SP);
	mXPUSHu((UV) _min_chunk_size);
	PUTBACK;
	nelts = python_call_pv("picviz_get_next_chunk", G_ARRAY);
	SPAGAIN;

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
		for (int j = 0; j < av_len(av); j++) {
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

	// Clear PYTHON stack
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

