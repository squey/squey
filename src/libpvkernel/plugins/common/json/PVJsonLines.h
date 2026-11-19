/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __RUSH_PVJSONLINES_H__
#define __RUSH_PVJSONLINES_H__

#include <QString>
#include <QStringList>

#include <cstddef>
#include <memory>
#include <vector>

namespace PVCore
{
class PVStreamingDecompressor;
} // namespace PVCore

namespace PVRush
{

/**
 * The lines of a file, read by blocks, which a parser can read in place. A file whose extension
 * is one of compressions() is decompressed as it is read.
 */
class PVJsonLines
{
  public:
	/**
	 * @throw std::runtime_error if the file @p path cannot be opened
	 */
	explicit PVJsonLines(const QString& path);
	~PVJsonLines();

  public:
	/** The extensions of the compressions of the files that are decompressed, such as "gz". */
	static QStringList compressions();

	/**
	 * The next line, without its end of line and ended by a null character, or nullptr after the
	 * last one. It stays valid and writable until the next call. The byte order mark of UTF-8 is
	 * left out of the first line.
	 *
	 * @throw std::runtime_error if the file cannot be read or decompressed
	 */
	char* next();

	/** The size of the line next() returned last. */
	size_t length() const { return _length; }

	/**
	 * The bytes of the file, as it is on disk, that the lines next() returned come from: those of
	 * the lines, ends of lines included, if the file is not compressed, and else the share of the
	 * bytes that its decompression read which the lines have of the data it gave. It never
	 * decreases, and counts all of the file once next() returned nullptr.
	 */
	size_t bytes_read() const { return _bytes_read; }

	/** The number of the line next() returned last, from 1. */
	size_t line_number() const { return _line_number; }

  private:
	QString _path;
	/** Through a pointer, for the <windows.h> that its header includes to stay out of this one. */
	std::unique_ptr<PVCore::PVStreamingDecompressor> _decompressor;
	/** Holds a byte more than the data read, for the null character after the last line. */
	std::vector<char> _buffer;
	/** The range of the data read that next() did not return yet, in _buffer. */
	size_t _begin = 0;
	size_t _end = 0;
	/** The size of the start of this range, which holds no end of line. */
	size_t _searched = 0;
	bool _at_end = false;
	size_t _length = 0;
	size_t _line_number = 0;
	/** Those of the lines next() returned, ends of lines included. */
	size_t _line_bytes = 0;
	/** Those of the data that the decompression gave, and those of the file it read for them. */
	size_t _data_bytes = 0;
	size_t _file_bytes = 0;
	size_t _bytes_read = 0;
};

} // namespace PVRush

#endif // __RUSH_PVJSONLINES_H__
