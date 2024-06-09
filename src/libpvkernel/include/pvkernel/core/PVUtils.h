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

#ifndef PVCORE_PVUTILS_H
#define PVCORE_PVUTILS_H

#include <stddef.h>
#include <string>
#include <QByteArray>
#include <QDataStream>
#include <QString>
#include <QIODevice>
#include <vector>

namespace PVCore
{

/**
 * Replace `from` with `to` in `init`.
 */
std::string&
replace(std::string& init, std::string const& from, std::string const& to, size_t pos = 0);

/**
 * Return the content of a file path as string
 */
std::string file_content(const std::string& file_path);

/**
 * Execute a command and return the content of the standard output
 */
std::string exec_cmd(const char* cmd);

template <typename T>
QString serialize_base64(const T& container)
{
	QByteArray byteArray;
	QDataStream out(&byteArray, QIODevice::WriteOnly);
	out << container;

	return QString(byteArray.toBase64());
}

template <typename T>
T deserialize_base64(const QString& str)
{
	T result;
	QByteArray byteArray = QByteArray::fromBase64(str.toUtf8());
	QDataStream in(&byteArray, QIODevice::ReadOnly);
	in >> result;

	return result;
}

template <typename It>
std::string join(It it_begin, It it_end, const std::string& separator)
{
	std::string ret;

	if (it_begin != it_end) {
		ret = *it_begin;
		++it_begin;
	}

	for (; it_begin != it_end; ++it_begin) {
		ret += separator;
		ret += *it_begin;
	}

	return ret;
}

void remove_common_folders(std::vector<std::string>& paths);

size_t available_memory();

} // namespace PVCore

#endif /* PVCORE_PVUTILS_H */
