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

#ifndef __SQUEY_PVPYTHONCOLUMNS__
#define __SQUEY_PVPYTHONCOLUMNS__

#include <stdexcept>
#include <string>
#include <vector>

namespace Squey
{
namespace PVPythonColumns
{

/**
 * Pick among the columns a name matched, or say why none was picked.
 *
 * The rule everything reading columns by name follows: a name may be carried
 * by more than one, and a position tells them apart. What each caller collects
 * is its own business -- a source, a result and the axes of a view do not
 * index the same thing -- but what is answered when the name is unknown, or
 * when the position runs past the namesakes, is one sentence written once.
 */
template <typename Index>
Index pick(const std::vector<Index>& matching, const std::string& column_name, size_t position)
{
	if (matching.empty()) {
		throw std::domain_error(std::string("No column named \"") + column_name + "\"");
	}
	if (position >= matching.size()) {
		throw std::domain_error(std::string("The count of column named \"") + column_name +
		                        "\" is <= " + std::to_string(position));
	}
	return matching[position];
}

} // namespace PVPythonColumns
} // namespace Squey

#endif // __SQUEY_PVPYTHONCOLUMNS__
