#include <pvkernel/rush/PVInputFile.h>
#include <fstream>
#include <errno.h>

PVRush::PVInputFile::PVInputFile(const char* path) :
	_path(path)
{
#ifdef WIN32
	_hfile = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (_hfile == INVALID_HANDLE_VALUE) {
		int err = GetLastError();
#else
	_file.open(path, std::ifstream::in);
	if (_file.fail()) {
		int err = errno;
#endif
		PVLOG_ERROR("Unable to open %s.\n", path);
		throw PVInputFileOpenException(path, err);
	}
}

PVRush::PVInputFile::~PVInputFile()
{
#ifdef WIN32
	CloseHandle(_hfile);
#else
	_file.close();
#endif
}

size_t PVRush::PVInputFile::operator()(char* buffer, size_t n)
{
	size_t ret;
#ifdef WIN32
	DWORD ret_win;
	if (!ReadFile(_hfile, buffer, n, &ret_win, NULL)) {
		ret_win = 0;
	}
	ret = ret_win;
#else
	_file.read(buffer, n);
	ret = _file.gcount();
#endif
	return ret;
}

PVRush::input_offset PVRush::PVInputFile::current_input_offset()
{
#ifdef WIN32
	LONG highPos;
	LONG lowPos = SetFilePointer(_hfile, 0, &highPos, FILE_CURRENT);
	return ((uint64_t)highPos << 32) | ((uint64_t)lowPos);
#else
	return _file.tellg();
#endif
}

void PVRush::PVInputFile::seek_begin()
{
#ifdef WIN32
	seek(0);
#else
	_file.clear();
	_file.seekg(0);
#endif
}

bool PVRush::PVInputFile::seek(input_offset off)
{
#ifdef WIN32
	LONG highPos = (LONG)(off>>32);
	return SetFilePointer(_hfile, off & 0x00000000FFFFFFFF, &highPos, FILE_BEGIN) != INVALID_SET_FILE_POINTER;
#else
	_file.clear();
	_file.seekg(off);
	return _file.good();
#endif
}

QString PVRush::PVInputFile::human_name()
{
	return QString(_path.c_str());
}

IMPL_INPUT(PVRush::PVInputFile)

