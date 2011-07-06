#ifndef PVINPUTFILE_FILE_H
#define PVINPUTFILE_FILE_H

#include <pvcore/general.h>
#include <pvrush/PVInput.h>
#include <fstream>
#include <string>

namespace PVRush {

class LibExport PVInputFile : public PVInput {
public:
	PVInputFile(const char* path);
	PVInputFile(const PVInputFile &org);
	~PVInputFile();
public:
	size_t operator()(char* buffer, size_t n);
	virtual input_offset current_input_offset();
	virtual void seek_begin();
	virtual QString human_name();
protected:
	std::ifstream _file;
	std::string _path;

	CLASS_INPUT(PVRush::PVInputFile)
};

}


#endif
