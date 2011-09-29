#ifndef PVINPUTFILE_FILE_H
#define PVINPUTFILE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInput.h>
#include <fstream>
#include <string>

#ifdef WIN32
#include <windows.h>
#endif

namespace PVRush {

class LibKernelDecl PVInputFile : public PVInput {
public:
	PVInputFile(const char* path);
	~PVInputFile();
private:
	PVInputFile(const PVInputFile &org) { assert(false); }
public:
	size_t operator()(char* buffer, size_t n);
	virtual input_offset current_input_offset();
	virtual void seek_begin();
	virtual bool seek(input_offset off);
	virtual QString human_name();
protected:
#ifdef WIN32
	HANDLE _hfile;
#else
	std::ifstream _file;
#endif
	std::string _path;

	CLASS_INPUT(PVRush::PVInputFile)
};

class PVInputFileOpenException: public PVInputException {
public:
	PVInputFileOpenException(const char* path, int err) :
		_path(path),
		_err(err)
	{
		_what = "Unable to open file ";
		_what += _path;
		_what += ": ";
#ifdef WIN32
		char* buffer;
		if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, err, 0, (LPSTR) &buffer, 0, NULL) != 0) {
			_what += buffer;
		}
		else {
			_what += "unable to get error message";
		}
		LocalFree(buffer);
#else
		_what += strerror(err);
#endif
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
