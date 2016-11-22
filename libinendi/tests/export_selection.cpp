/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVExporter.h>
#include <pvkernel/core/inendi_assert.h>

#include <inendi/PVSelection.h>
#include <inendi/PVScene.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>

#include <cstdlib>
#include <iostream>

#include <QFile>

#include "common.h"

static constexpr const PVCore::PVExporter::CompressionType compression_type =
    PVCore::PVExporter::CompressionType::GZ;

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	pvtest::TestEnv env(argv[1], argv[2], 1, pvtest::ProcessUntil::View);

	bool delete_nraw_parent_dir = false;
	QDir nraw_dir(QString::fromStdString(PVRush::PVNraw::default_tmp_path));
	if (!nraw_dir.exists()) {
		nraw_dir.mkdir(QString::fromStdString(PVRush::PVNraw::default_tmp_path));
		delete_nraw_parent_dir = true;
	}

	Inendi::PVView* view = env.root.current_view();

	// Export selection to temporary file
	Inendi::PVSelection sel(view->get_row_count());
	sel.select_all();

	char temp_pattern[] = "/tmp/fileXXXXXX";
	close(mkstemp(temp_pattern));
	std::remove(temp_pattern);
	std::string output_file =
	    std::string(temp_pattern) + PVCore::PVExporter::extension(compression_type);

	PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const PVCore::PVColumnIndexes& col_indexes =
	    view->get_parent<Inendi::PVSource>().get_format().get_axes_comb();

	PVCore::PVExporter::export_func export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };

	auto start = std::chrono::system_clock::now();

	PVCore::PVExporter exp(output_file, sel, col_indexes, nraw.row_count(), export_func,
	                       compression_type);
	exp.export_rows(0);
	exp.wait_finished();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	std::string cmd = "gunzip " + output_file;
	int result = system(cmd.c_str());
	PV_VALID(result, 0);
	std::string uncompressed_file = output_file.substr(0, output_file.find_last_of("."));
	bool same_content = PVRush::PVUtils::files_have_same_content(argv[1], uncompressed_file);
	std::cout << std::endl << argv[1] << " - " << uncompressed_file << std::endl;
	PV_ASSERT_VALID(same_content);
	std::remove(uncompressed_file.c_str());
#endif // INSPECTOR_BENCH

	std::remove(output_file.c_str());

	// Remove nraw folder
	PVCore::PVDirectory::remove_rec(delete_nraw_parent_dir ? nraw_dir.path()
	                                                       : QString::fromStdString(nraw.dir()));
}
