/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QCryptographicHash>
#include <QFile>
#include <QString>

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/rush/PVUtils.h>

#include <fstream>

/*
QString PVRush::PVUtils::generate_key_from_axes_values(PVCore::PVAxesIndexType const& axes, PVRush::PVNraw::const_nraw_table_line const& values)
{
	QString ret;
	PVCore::PVAxesIndexType::const_iterator it;
	for (it = axes.begin(); it != axes.end(); it++) {
		ret.append(values[*it].get_qstr());
	}
	return ret;
}
*/

const QByteArray PVRush::PVUtils::get_file_checksum(const QString& path)
{
	QFile file;
	file.setFileName(path);
	if (!file.open(QIODevice::ReadOnly)) {
		return QByteArray();
	}
	QByteArray data = file.readAll();
	file.close();

	return QCryptographicHash::hash(data, QCryptographicHash::Md5).toHex();
}

bool PVRush::PVUtils::files_have_same_content(const QString& path1, const QString& path2)
{
	return get_file_checksum(path1) == get_file_checksum(path2);
}

void PVRush::PVUtils::sort_file(const char* input_file, const char* output_file /*= nullptr*/)
{
	std::ifstream fin(input_file);
	std::vector<std::string> array;

	while (true) {
		std::string s;
		getline(fin, s);
		if (fin.eof()) {
			break;
		}
		array.emplace_back(s);
	}
	fin.close();

	std::sort(array.begin(), array.end());

	std::ofstream fout(output_file ? output_file : input_file);
	std::copy(array.begin(), array.end(), std::ostream_iterator<std::string>(fout, "\n"));
	fout.close();
}

static std::string& replace(std::string& init, std::string const& from, std::string const& to)
{
        size_t pos = 0;
        while ((pos = init.find(from, pos)) != std::string::npos) {
                init.replace(pos, from.size(), to);
                // Advance to avoid replacing the same character again (case of " for example)
                pos += to.size();
        }
        return init;
}

std::string PVRush::PVUtils::safe_export(std::string str, const std::string& quote_char)
{
	static std::string escaped_quote("\\" + quote_char);

        return quote_char + replace(replace(replace(str, "\n", "\\n"), "\r", "\\r"), quote_char, escaped_quote) + quote_char;
}

void PVRush::PVUtils::safe_export(QStringList& str_list, const std::string& quote_char)
{
	for (QString& str : str_list) {
		str = QString::fromStdString(safe_export(str.toStdString(), quote_char));
	}
}
