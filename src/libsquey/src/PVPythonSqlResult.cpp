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

#include <squey/PVPythonSqlResult.h>

#include <stdexcept>
#include <utility>

Squey::PVPythonSqlResult::PVPythonSqlResult(std::vector<PVDuckDBQuery::ResultColumn> columns)
    : _columns(std::move(columns))
{
}

size_t Squey::PVPythonSqlResult::row_count() const
{
	// Every column holds as many rows; a result with no column has no rows.
	return _columns.empty() ? 0 : _columns.front().valid.size();
}

size_t Squey::PVPythonSqlResult::column_count() const
{
	return _columns.size();
}

const Squey::PVDuckDBQuery::ResultColumn& Squey::PVPythonSqlResult::at(size_t column_index) const
{
	if (column_index >= _columns.size()) {
		throw std::out_of_range("Out of range column index");
	}
	return _columns[column_index];
}

size_t Squey::PVPythonSqlResult::index_of(const std::string& column_name, size_t position) const
{
	size_t seen = 0;
	for (size_t i = 0; i < _columns.size(); i++) {
		if (_columns[i].name == column_name && seen++ == position) {
			return i;
		}
	}
	if (seen == 0) {
		throw std::domain_error(std::string("No column named \"") + column_name + "\"");
	}
	throw std::domain_error(std::string("The count of column named \"") + column_name +
	                        "\" is <= " + std::to_string(position));
}

std::string Squey::PVPythonSqlResult::column_name(size_t column_index) const
{
	return at(column_index).name;
}

std::string Squey::PVPythonSqlResult::column_type(size_t column_index) const
{
	return at(column_index).type;
}

std::string Squey::PVPythonSqlResult::column_type(const std::string& column_name,
                                                  size_t position) const
{
	return column_type(index_of(column_name, position));
}

pybind11::array Squey::PVPythonSqlResult::column(size_t column_index) const
{
	const PVDuckDBQuery::ResultColumn& held = at(column_index);
	switch (held.kind) {
	case PVDuckDBQuery::ResultColumn::Kind::Integer:
		return pybind11::array_t<int64_t>(held.integers.size(), held.integers.data());
	case PVDuckDBQuery::ResultColumn::Kind::Real:
		return pybind11::array_t<double>(held.reals.size(), held.reals.data());
	default:
		break;
	}
	return pybind11::array(pybind11::cast(held.texts));
}

pybind11::array Squey::PVPythonSqlResult::column(const std::string& column_name,
                                                 size_t position) const
{
	return column(index_of(column_name, position));
}

pybind11::array Squey::PVPythonSqlResult::valid(size_t column_index) const
{
	const PVDuckDBQuery::ResultColumn& held = at(column_index);
	pybind11::array array(pybind11::dtype("bool"), held.valid.size());
	std::copy(held.valid.begin(), held.valid.end(),
	          static_cast<uint8_t*>(array.request().ptr));
	return array;
}

pybind11::array Squey::PVPythonSqlResult::valid(const std::string& column_name,
                                                size_t position) const
{
	return valid(index_of(column_name, position));
}
