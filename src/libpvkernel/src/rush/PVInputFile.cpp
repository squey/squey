//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVInput.h> // for IMPL_INPUT
#include <pvkernel/rush/PVInputFile.h>
#include <qstring.h>
#include <stddef.h>
#include <stdint.h>
#include <fstream>
#include <memory>

#include "pvkernel/core/PVStreamingCompressor.h"

PVRush::PVInputFile::PVInputFile(const char* path) : _path(path), _decompressor(_path)
{
}

PVRush::PVInputFile::~PVInputFile()
= default;

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
	return {_path.c_str()};
}

uint64_t PVRush::PVInputFile::file_size()
{
	return std::ifstream(_path, std::ifstream::ate | std::ifstream::binary).tellg();
}

IMPL_INPUT(PVRush::PVInputFile)
