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

#ifndef __PVRUSH_PVROWFILTER_H__
#define __PVRUSH_PVROWFILTER_H__

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ArrowArrayStream;

namespace PVRush
{

/**
 * The rows of Arrow streams that a SQL condition holds for: the condition of a WHERE clause, over
 * columns named as those of the streams.
 *
 * The library of the views registers the engine, as it links the SQL one: what reads files can
 * then filter their rows without linking it.
 */
class PVRowFilter
{
  public:
	using factory_t = std::function<std::unique_ptr<PVRowFilter>(const std::string& condition)>;

	/** The engine of the filters that make() returns. */
	static void set_factory(factory_t factory);

	/**
	 * A filter of the rows that @p condition holds for.
	 *
	 * @throw std::runtime_error if no engine is registered
	 */
	static std::unique_ptr<PVRowFilter> make(const std::string& condition);

  public:
	virtual ~PVRowFilter() = default;

	/**
	 * The values that the column @p index_column of @p rows takes in the rows that the condition
	 * holds for, in no particular order. Takes @p rows over, and releases it.
	 *
	 * @throw std::runtime_error with the message of the engine, if the condition is not one over
	 * the columns of @p rows
	 */
	virtual std::vector<int64_t> matching_rows(ArrowArrayStream& rows,
	                                           const std::string& index_column) = 0;
};

} // namespace PVRush

#endif // __PVRUSH_PVROWFILTER_H__
