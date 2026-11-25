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

#ifndef __RUSH_PVARROWROWFILTER_H__
#define __RUSH_PVARROWROWFILTER_H__

#include <pvkernel/rush/PVRowFilter.h>

#include <arrow/type_fwd.h>

#include <memory>
#include <string>

namespace PVRush
{

/**
 * Keeps the rows of Arrow tables that a SQL condition holds for, which reads their columns by
 * their names.
 */
class PVArrowRowFilter
{
  public:
	/**
	 * @throw std::runtime_error if no engine filters the rows
	 */
	explicit PVArrowRowFilter(const std::string& condition);

  public:
	/**
	 * The indexes of the rows of @p table that the condition holds for, in their order, as an
	 * array of int64.
	 *
	 * @throw std::runtime_error if the condition is not one over the columns of @p table
	 */
	std::shared_ptr<arrow::Array> matching_rows(const std::shared_ptr<arrow::Table>& table);

	/**
	 * The rows of @p table that the condition holds for, in their order.
	 *
	 * @throw std::runtime_error as matching_rows() does
	 */
	std::shared_ptr<arrow::Table> filter(const std::shared_ptr<arrow::Table>& table);

	/**
	 * What is wrong with @p condition, as a condition over the columns of @p schema, or nothing.
	 */
	static std::string error_of(const std::string& condition,
	                            const std::shared_ptr<arrow::Schema>& schema);

  private:
	std::unique_ptr<PVRowFilter> _filter;
};

} // namespace PVRush

#endif // __RUSH_PVARROWROWFILTER_H__
