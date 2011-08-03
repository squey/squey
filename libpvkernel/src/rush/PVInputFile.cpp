#include <pvkernel/rush/PVInputFile.h>
#include <fstream>
#include <errno.h>

PVRush::PVInputFile::PVInputFile(const char* path) :
	_path(path)
{
	// TODO: check for failure !
	_file.open(path, std::ifstream::in);
	if (_file.fail()) {
		int err = errno;
		PVLOG_ERROR("Unable to open %s.\n", path);
		throw PVInputFileOpenException(path, err);
	}
}

PVRush::PVInputFile::PVInputFile(const PVInputFile &org)
{
	_path = org._path;
	_file.open(org._path.c_str(), std::ifstream::in);
}

PVRush::PVInputFile::~PVInputFile()
{
	_file.close();
}

size_t PVRush::PVInputFile::operator()(char* buffer, size_t n)
{
	_file.read(buffer, n);
	size_t ret = _file.gcount();
	return ret;
}

PVRush::PVInputFile::input_offset PVRush::PVInputFile::current_input_offset()
{
	return _file.tellg();
}

void PVRush::PVInputFile::seek_begin()
{
	_file.clear();
	_file.seekg(0);
}

QString PVRush::PVInputFile::human_name()
{
	return QString(_path.c_str());
}

IMPL_INPUT(PVRush::PVInputFile)
