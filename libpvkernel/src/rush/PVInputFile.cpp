/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVInputFile.h>
#include <fstream>
#include <errno.h>

PVRush::PVInputFile::PVInputFile(const char* path) : _path(path)
{
	_file.open(path, std::ifstream::in);
	if (_file.fail()) {
		int err = errno;
		PVLOG_ERROR("Unable to open %s.\n", path);
		throw PVInputFileOpenException(path, err);
	}
}

PVRush::PVInputFile::~PVInputFile()
{
	release();
}

void PVRush::PVInputFile::release()
{
	if (_file.is_open()) {
		_file.close();
	}
}

size_t PVRush::PVInputFile::operator()(char* buffer, size_t n)
{
	_file.read(buffer, n);
	size_t ret = _file.gcount();
	return ret;
}

PVRush::input_offset PVRush::PVInputFile::current_input_offset()
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

uint64_t PVRush::PVInputFile::file_size()
{
	uint64_t cur_off = _file.tellg();
	_file.clear();
	_file.seekg(0, std::ios::end);
	uint64_t size = _file.tellg();
	_file.seekg(cur_off, std::ios::beg);
	return size;
}

IMPL_INPUT(PVRush::PVInputFile)
