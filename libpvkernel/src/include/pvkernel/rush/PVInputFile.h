/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef PVINPUTFILE_FILE_H
#define PVINPUTFILE_FILE_H

#include <pvkernel/core/PVStreamingCompressor.h>
#include <pvkernel/rush/PVInput.h> // for PVInputException, etc

#include <atomic>
#include <cstddef> // for size_t
#include <cstdint> // for uint64_t
#include <cstring> // for strerror
#include <fstream> // for ifstream
#include <string>  // for allocator, operator+, etc
#include <thread>

#include <QString>

namespace PVRush
{

class PVInputFile : public PVInput
{
  public:
	explicit PVInputFile(const char* path);
	PVInputFile(const PVInputFile& /*org*/) = delete;
	~PVInputFile() override;

  public:
	chunk_sizes_t operator()(char* buffer, size_t n) override;
	void cancel() override;
	void seek_begin() override;
	QString human_name() override;

  public:
	// File specific
	uint64_t file_size();
	std::string file_path() const { return _path; }

  private:
	std::string _path;
	PVCore::PVStreamingDecompressor _decompressor;

	CLASS_INPUT(PVRush::PVInputFile)
};

class PVInputFileOpenException : public PVInputException
{
  public:
	PVInputFileOpenException(std::string const& path, int err)
	    : PVInputException("Unable to open file " + path + ": " + strerror(err))
	{
	}
};
} // namespace PVRush

#endif
