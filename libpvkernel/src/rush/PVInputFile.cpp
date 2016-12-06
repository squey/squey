/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVInput.h> // for IMPL_INPUT
#include <pvkernel/rush/PVInputFile.h>

#include <pvkernel/core/PVLogger.h> // for PVLOG_ERROR
#include <pvkernel/core/PVArchive.h>

#include <cerrno>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>

PVRush::PVInputFile::PVInputFile(const char* path) : _path(path), _decompressor(_path)
{
}

PVRush::PVInputFile::~PVInputFile()
{
}

PVRush::PVInputFile::chunk_sizes_t PVRush::PVInputFile::operator()(char* buffer, size_t n)
{
	try {
		return _decompressor.read(buffer, n);
	} catch (const PVCore::PVStreamingDecompressorError& e) {
		throw PVInputException(e.what());
	}
}

void PVRush::PVInputFile::cancel()
{
	_decompressor.cancel();
}

void PVRush::PVInputFile::seek_begin()
{
	_decompressor.reset();
}

QString PVRush::PVInputFile::human_name()
{
	return QString(_path.c_str());
}

uint64_t PVRush::PVInputFile::file_size()
{
	return std::ifstream(_path, std::ifstream::ate | std::ifstream::binary).tellg();
}

IMPL_INPUT(PVRush::PVInputFile)
