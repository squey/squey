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

#ifndef PVKERNEL_CORE_PVSERIALIZEDSOURCE_HPP
#define PVKERNEL_CORE_PVSERIALIZEDSOURCE_HPP

#include <vector>
#include <string>

// TODO : input_desc is "designed" to be use for files. To make it works with others inputs, we
// should make PVSerializedSource a template class with the description type as template argument or
// create smart parser on this string.

namespace PVCore
{
struct PVSerializedSource {
	std::vector<std::vector<std::string>> input_desc;
	std::string sc_name;
	std::string format_name;
	std::string format_path;

	bool need_credential() const { return sc_name != "text_file"; }
};
} // namespace PVCore

#endif
