#include <pvkernel/rush/PVOutputFile.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <stdio.h>

PVRush::PVOutputFile::PVOutputFile(const char* path)
{
	_file = fopen(path, "w");
	if (_file == NULL)
		PVLOG_ERROR("PVOutputFile:: unable to open %s\n", path);
}

PVRush::PVOutputFile::~PVOutputFile()
{
	/*if (_file != NULL) {
		fclose(_file);
		_file = NULL;
	}*/
}

void PVRush::PVOutputFile::operator()(PVCore::PVChunk* out)
{
	// Write fields separated by '\t'
	PVCore::list_elts const& le = out->c_elements();
	PVCore::list_elts::const_iterator it,ite;
	ite = le.end();
	for (it = le.begin(); it != ite; it++) {
		PVCore::PVElement const& elt = *(*it);
		if (!elt.valid())
			continue;
		PVCore::list_fields const& lf = elt.c_fields();
		PVCore::list_fields::const_iterator itf,itef;
		itef = lf.end();
		for (itf = lf.begin(); itf != itef; itf++) {
			PVCore::PVField const& field = *itf;
			if (!field.valid())
				continue;
			const QByteArray flocal = field.get_qstr().toLocal8Bit();
			fwrite(flocal.data(), flocal.size(), 1, _file);
			fputc('\t', _file);
		}
		fputc('\n', _file);
	}
	out->free();
}
