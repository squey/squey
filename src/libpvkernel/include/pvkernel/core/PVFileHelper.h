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

#ifndef PVCORE_PVFILEHELPER_H
#define PVCORE_PVFILEHELPER_H

#include <QString>

namespace PVCore
{

struct PVFileHelper {

	static bool is_already_opened(const char* file_name);

	/**
	 * The suffixes of the name of the file @p path, after its first dot and in lower case, but
	 * for the numbers and the dates that the rotation of files adds: "eve.json.2.gz",
	 * "eve.json-20261119.gz" and "eve.json.gz.1" have "json.gz", and "report.2026.01.csv" has
	 * "csv".
	 */
	static QString suffixes_without_rotation(const QString& path);
};
} // namespace PVCore

#endif // PVCORE_PVFILEHELPER_H
