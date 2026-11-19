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

#include <cstring>
#include <stdexcept>
#include <string>

namespace
{

constexpr const size_t BLOCK_SIZE = 4 * 1024 * 1024;
constexpr const char BYTE_ORDER_MARK[] = "\xEF\xBB\xBF";

} // namespace

PVRush::PVJsonLines::PVJsonLines(const QString& path) : _file(path)
{
	if (not _file.open(QIODevice::ReadOnly)) {
		throw std::runtime_error("Unable to read '" + path.toStdString() +
		                         "': " + _file.errorString().toStdString());
	}
	_buffer.resize(BLOCK_SIZE + 1);
}

char* PVRush::PVJsonLines::next()
{
	while (true) {
		char* begin = _buffer.data() + _begin;
		char* end = _buffer.data() + _end;
		char* line_end = static_cast<char*>(std::memchr(begin, '\n', size_t(end - begin)));
		if (line_end != nullptr or (_at_end and begin != end)) {
			const size_t consumed = size_t((line_end != nullptr ? line_end + 1 : end) - begin);
			if (line_end == nullptr) {
				line_end = end;
			}
			if (line_end != begin and line_end[-1] == '\r') {
				line_end--;
			}
			*line_end = '\0';
			_begin += consumed;
			_bytes_read += consumed;
			_line_number++;
			if (_line_number == 1 and std::strncmp(begin, BYTE_ORDER_MARK, 3) == 0) {
				begin += 3;
			}
			_length = size_t(line_end - begin);
			return begin;
		}
		if (_at_end) {
			return nullptr;
		}

		// The start of the line goes first, for the rest of it to follow.
		std::memmove(_buffer.data(), begin, size_t(end - begin));
		_end -= _begin;
		_begin = 0;
		if (_end + BLOCK_SIZE + 1 > _buffer.size()) {
			_buffer.resize(_end + BLOCK_SIZE + 1);
		}
		const qint64 read = _file.read(_buffer.data() + _end, BLOCK_SIZE);
		if (read < 0) {
			throw std::runtime_error("Unable to read '" + _file.fileName().toStdString() +
			                         "': " + _file.errorString().toStdString());
		}
		_end += size_t(read);
		_at_end = read == 0;
	}
}
