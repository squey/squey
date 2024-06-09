//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/PVUtils.h>
#include <qcontainerfwd.h>
#include <qlist.h>
#include <QString>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

bool PVRush::PVUtils::files_have_same_content(const std::string& path1, const std::string& path2)
{
	std::ifstream ifs1(path1);
	std::ifstream ifs2(path2);

	if (not ifs1.good() or not ifs2.good()) {
		return false;
	}

	auto res = std::mismatch(std::istreambuf_iterator<char>(ifs1), std::istreambuf_iterator<char>(),
	                         std::istreambuf_iterator<char>(ifs2));

	return res.first == std::istreambuf_iterator<char>() &&
	       res.second == std::istreambuf_iterator<char>();
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

std::string PVRush::PVUtils::safe_export(std::string str,
                                         const std::string& sep_char,
                                         const std::string& quote_char)
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

void PVRush::PVUtils::safe_export(QStringList& str_list,
                                  const std::string& sep_char,
                                  const std::string& quote_char)
{
	for (QString& str : str_list) {
		str = QString::fromStdString(safe_export(str.toStdString(), sep_char, quote_char));
	}
}
