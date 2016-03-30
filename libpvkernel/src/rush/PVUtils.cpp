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
#include <pvkernel/core/PVUtils.h>

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

std::string PVRush::PVUtils::safe_export(std::string str, const std::string& sep_char, const std::string& quote_char)
{
	static std::string escaped_quote("\\" + quote_char);

	bool do_quote = false;

	if (str.find(sep_char) != std::string::npos) {
		do_quote = true;
	}
	if (str.find(quote_char) != std::string::npos) {
		do_quote = true;
		PVCore::replace(str, quote_char, escaped_quote);
	}
	if (do_quote) {
		str.append(quote_char);
		str.insert(0, quote_char);
	}

	return str;
}

void PVRush::PVUtils::safe_export(QStringList& str_list, const std::string& sep_char, const std::string& quote_char)
{
	for (QString& str : str_list) {
		str = QString::fromStdString(safe_export(str.toStdString(), sep_char, quote_char));
	}
}
