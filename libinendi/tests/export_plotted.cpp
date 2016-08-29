/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */
#include "common.h"

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/PVExporter.h>

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

	std::ofstream stream(output_tmp_file);

	Inendi::PVView* view = env.root.current_view();
	PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const PVCore::PVColumnIndexes& col_indexes =
	    view->get_parent<Inendi::PVSource>().get_format().get_axes_comb();

	PVCore::PVExporter::export_func export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) {
		    return view->get_parent<Inendi::PVPlotted>().export_line(row, cols, sep, quote);
		};

	Inendi::PVSelection sel(nraw.get_row_count());
	sel.select_all();
	PVCore::PVExporter exp(stream, sel, col_indexes, nraw.get_row_count(), export_func);

	auto start = std::chrono::system_clock::now();

	exp.export_rows(0);

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
