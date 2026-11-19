//
// MIT License
//
// © Squey, 2026
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

#include "PVJsonLines.h"

#include <pvkernel/core/PVStreamingCompressor.h>

#include <QFile>
#include <QFileInfo>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace
{

constexpr const size_t BLOCK_SIZE = 4 * 1024 * 1024;
constexpr const char BYTE_ORDER_MARK[] = "\xEF\xBB\xBF";

} // namespace

PVRush::PVJsonLines::PVJsonLines(const QString& path)
    : _path(path)
    , _decompressor(std::make_unique<PVCore::PVStreamingDecompressor>(path.toStdString()))
{
	// The decompression only opens the file once read, and does not tell why it cannot.
	QFile file(path);
	if (not file.open(QIODevice::ReadOnly)) {
		throw std::runtime_error("Unable to read '" + path.toStdString() +
		                         "': " + file.errorString().toStdString());
	}
	_buffer.resize(BLOCK_SIZE + 1);
}

PVRush::PVJsonLines::~PVJsonLines() = default;

QStringList PVRush::PVJsonLines::compressions()
{
	QStringList extensions;
	for (const std::string& extension : PVCore::PVStreamingDecompressor::supported_extensions()) {
		extensions << QString::fromStdString(extension);
	}
	return extensions;
}

char* PVRush::PVJsonLines::next()
{
	while (true) {
		char* begin = _buffer.data() + _begin;
		char* end = _buffer.data() + _end;
		char* line_end = static_cast<char*>(
		    std::memchr(begin + _searched, '\n', size_t(end - begin) - _searched));
		if (line_end != nullptr or (_at_end and begin != end)) {
			_searched = 0;
			const size_t consumed = size_t((line_end != nullptr ? line_end + 1 : end) - begin);
			if (line_end == nullptr) {
				line_end = end;
			}
			if (line_end != begin and line_end[-1] == '\r') {
				line_end--;
			}
			*line_end = '\0';
			_begin += consumed;
			_line_bytes += consumed;
			_line_number++;
			// The data of a compressed file stands in proportion for the bytes it was read from.
			const size_t share =
			    _file_bytes == _data_bytes
			        ? _line_bytes
			        : size_t(double(_file_bytes) * (double(_line_bytes) / double(_data_bytes)));
			_bytes_read = std::max<size_t>(_bytes_read, share);
			if (_line_number == 1 and std::strncmp(begin, BYTE_ORDER_MARK, 3) == 0) {
				begin += 3;
			}
			_length = size_t(line_end - begin);
			return begin;
		}
		if (_at_end) {
			// A decompression can stop reading a file once it gave all of its data.
			_bytes_read =
			    std::max<size_t>({_bytes_read, _file_bytes, size_t(QFileInfo(_path).size())});
			return nullptr;
		}

		// A pipe gives the data of a long line in parts of 64 kB: what was searched of it is not
		// searched again, nor moved, for each part, which would take a time growing as the square
		// of its length.
		_searched = size_t(end - begin);
		// The start of the line goes first, for the rest of it to follow.
		if (_begin > 0) {
			std::memmove(_buffer.data(), begin, size_t(end - begin));
			_end -= _begin;
			_begin = 0;
		}
		if (_end + BLOCK_SIZE + 1 > _buffer.size()) {
			_buffer.resize(_end + BLOCK_SIZE + 1);
		}
		PVCore::PVStreamingDecompressor::chunk_sizes_t sizes;
		try {
			sizes = _decompressor->read(_buffer.data() + _end, BLOCK_SIZE);
		} catch (const PVCore::PVStreamingDecompressorError& e) {
			throw std::runtime_error("Unable to read '" + _path.toStdString() + "': " + e.what());
		}
		_end += sizes.first;
		_data_bytes += sizes.first;
		_file_bytes += sizes.second;
		_at_end = sizes.first == 0;
	}
}
