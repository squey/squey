/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QFile>
#include <QString>

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/rush/PVUtils.h>

#include <fstream>

bool PVRush::PVUtils::files_have_same_content(const std::string& path1, const std::string& path2)
{
	std::ifstream ifs1(path1);
	std::ifstream ifs2(path2);

	auto res = std::mismatch(std::istreambuf_iterator<char>(ifs1), std::istreambuf_iterator<char>(), std::istreambuf_iterator<char>(ifs2));
	std::cout << (res.first == std::istreambuf_iterator<char>()) << "/" << (res.second == std::istreambuf_iterator<char>()) << std::endl;

	return res.first == std::istreambuf_iterator<char>() && res.second == std::istreambuf_iterator<char>();
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
