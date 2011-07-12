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

class PVInputFileOpenException {
public:
	PVInputFileOpenException(const char* path, int err) :
		_path(path),
		_err(err)
	{
		_what = "Unable to open file ";
		_what += _path;
		_what += ": ";
		_what += strerror(err);
	}
public:
	inline int err() const { return _err; }
	inline std::string const& path() const { return _path; }
	std::string const& what() const { return _what; }
protected:
	std::string _path;
	int _err;
	std::string _what;
};

}


#endif
