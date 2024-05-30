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

#ifndef PVINPUT_FILE_H
#define PVINPUT_FILE_H

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <QString>

#include <pvkernel/rush/PVInput_types.h>

namespace PVRush
{

class PVInput
{
  public:
	using p_type = PVInput_p;
	using chunk_sizes_t = std::pair<size_t /*uncompressed*/, size_t /*compressed*/>;

  public:
	virtual ~PVInput() = default;

  public:
	// This method must read at most n bytes and put the result in buffer and returns the number of
	// bytes actually read.
	// It returns 0 if no more data is available
	virtual chunk_sizes_t operator()(char* buffer, size_t n) = 0;
	// Seek to the beggining of the input
	virtual void cancel() = 0;
	virtual void seek_begin() = 0;
	virtual QString human_name() = 0;
};

class PVInputException : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};
} // namespace PVRush

#define IMPL_INPUT(T)
#define CLASS_INPUT(T)

#endif
