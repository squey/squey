/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVUTILS_H
#define PVCORE_PVUTILS_H

#include <string>

#include <QByteArray>
#include <QDataStream>
#include <QString>

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

} // namespace PVCore

#endif /* PVCORE_PVUTILS_H */
