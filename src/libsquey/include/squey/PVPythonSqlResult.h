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

#ifndef __SQUEY_PVPYTHONSQLRESULT__
#define __SQUEY_PVPYTHONSQLRESULT__

#include <squey/PVDuckDBQuery.h>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <string>
#include <vector>

namespace Squey
{

/**
 * What a query gives back, read the way a source is read.
 *
 * The same words as a source for everything that is reading -- column_count,
 * column_name, column_type, column, valid, row_count -- so a function written
 * against one works on the other. What a source has and this has not is the
 * rest: it holds no scopes, since a result is not a set of rows of the source,
 * and nothing can be added to it, since it is a value rather than a place.
 */
class PVPythonSqlResult
{
  public:
	explicit PVPythonSqlResult(std::vector<PVDuckDBQuery::ResultColumn> columns);

  public:
	PYBIND11_EXPORT size_t row_count() const;
	PYBIND11_EXPORT size_t column_count() const;

	PYBIND11_EXPORT std::string column_name(size_t column_index) const;

	//! What DuckDB called the column -- BIGINT, VARCHAR, TIMESTAMP.
	PYBIND11_EXPORT std::string column_type(size_t column_index) const;
	PYBIND11_EXPORT std::string column_type(const std::string& column_name, size_t position) const;

	/**
	 * One column, as an array.
	 *
	 * Whole numbers come back as int64, real ones as double, and everything
	 * else as the text it prints as. Where a row has no value the array holds
	 * what its type reads as empty -- a zero, an empty string -- which is what
	 * valid() is there to tell apart.
	 */
	PYBIND11_EXPORT pybind11::array column(size_t column_index) const;
	PYBIND11_EXPORT pybind11::array column(const std::string& column_name, size_t position) const;

	//! Which rows of a column carry a value. See PVPythonSource::valid().
	PYBIND11_EXPORT pybind11::array valid(size_t column_index) const;
	PYBIND11_EXPORT pybind11::array valid(const std::string& column_name, size_t position) const;

  private:
	/**
	 * Which column a name stands for, @a position telling namesakes apart. A
	 * query names its own columns, and nothing stops it naming two alike.
	 */
	size_t index_of(const std::string& column_name, size_t position) const;

	const PVDuckDBQuery::ResultColumn& at(size_t column_index) const;

  private:
	std::vector<PVDuckDBQuery::ResultColumn> _columns;
};

} // namespace Squey

#endif // __SQUEY_PVPYTHONSQLRESULT__
