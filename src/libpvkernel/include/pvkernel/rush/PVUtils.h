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

#ifndef PVRUSH_PVUTILS_H
#define PVRUSH_PVUTILS_H

class QString;
#include <QStringList>
#include <QByteArray>

#include <pvkernel/rush/PVNraw.h>

#include <string.h>

namespace PVRush
{
namespace PVUtils
{
bool files_have_same_content(const std::string& path1, const std::string& path2);

/**
 * Alphabetically sort the lines of a text file
 *
 * @param input_file input file path
 * @param output_file output file path (inplace sort if not specified)
 */
void sort_file(const char* input_file, const char* output_file = nullptr);

std::string
safe_export(std::string str, const std::string& sep_char, const std::string& quote_char);
void safe_export(QStringList& str_list, const std::string& sep_char, const std::string& quote_char);
} // namespace PVUtils
} // namespace PVRush

#endif /* PVRUSH_PVUTILS_H */
