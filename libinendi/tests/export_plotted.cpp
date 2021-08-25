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

#include "common.h"

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/rush/PVCSVExporter.h>

#include <sys/stat.h>

#ifdef INSPECTOR_BENCH
constexpr size_t DUPL = 10000;
#else
constexpr size_t DUPL = 1;
#endif

int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " input_file output_file format" << std::endl;
		return 1;
	}

	const char* input_file = argv[1];
	const char* format = argv[3];

	pvtest::TestEnv env(input_file, format, DUPL, pvtest::ProcessUntil::View);

	std::string output_tmp_file = pvtest::get_tmp_filename();

	Inendi::PVView* view = env.root.current_view();
	PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const PVCore::PVColumnIndexes& col_indexes =
	    view->get_parent<Inendi::PVSource>().get_format().get_axes_comb();

	PVRush::PVCSVExporter::export_func_f export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) {
		    return view->get_parent<Inendi::PVPlotted>().export_line(row, cols, sep, quote);
		};

	Inendi::PVSelection sel(nraw.row_count());
	sel.select_all();

	auto start = std::chrono::system_clock::now();

	PVRush::PVCSVExporter exp(col_indexes, nraw.row_count(), export_func);
	exp.export_rows(output_tmp_file, sel);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	const char* output_file = argv[2];
	PV_VALID(PVCore::file_content(output_tmp_file), PVCore::file_content(output_file));
#endif

	std::remove(output_tmp_file.c_str());

	return 0;
}
