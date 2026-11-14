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

// The values the CSV splitter cuts out of lines holding quotes.

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "helpers.h"
#include "common.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

int main()
{
	// each line, and the three values it holds as the test helpers dump them
	const std::vector<std::pair<std::string, std::string>> lines{
	    {R"(1,"quoted, value",2)", R"('1','quoted, value','2')"},
	    // a quote nothing closes is part of the value
	    {R"(3,12" pipe,45.2)", R"('3','12" pipe','45.2')"},
	    {R"(4,5" x 7",8)", R"('4','5" x 7"','8')"},
	    // quotes inside a value that is not quoted still hold separators
	    {R"(9,key="a,b",10)", R"('9','key="a,b"','10')"},
	};

	const QString path_qs =
	    PVCore::mkstemp(PVRush::PVNrawCacheManager::nraw_dir() + "/splitter_csv_quotes_XXXXXX.csv");
	const std::string path = path_qs.toStdString();
	{
		std::ofstream out(path);
		for (const auto& line : lines) {
			out << line.first << '\n';
		}
	}

	pvtest::TestSplitter ts(path);

	PVFilter::PVFieldsSplitter::p_type splitter =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("csv");
	splitter->set_number_expected_fields(3);
	PVCore::PVArgumentList args = splitter->get_args();
	args["sep"] = QVariant(QChar(','));
	splitter->set_args(args);

	auto by_fields = std::make_unique<PVFilter::PVElementFilterByFields>();
	by_fields->add_filter(splitter);
	PVFilter::PVChunkFilterByElt chunk_filter{std::move(by_fields)};

	const std::string output = std::get<2>(ts.run_normalization(chunk_filter));
	std::ifstream in(output);
	for (const auto& line : lines) {
		std::string split;
		std::getline(in, split);
		PV_VALID(split, line.second, "line", line.first);
	}

	std::remove(output.c_str());
	std::remove(path.c_str());

	return 0;
}
