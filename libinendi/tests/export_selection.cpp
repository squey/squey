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
#include <inendi/PVSelection.h>
#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <inendi/PVMapping.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotting.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>
#include <cstdlib>
#include <iostream>
#include <QFile>
#include "common.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	pvtest::TestEnv env(argv[1], argv[2]);

	bool delete_nraw_parent_dir = false;
	QDir nraw_dir(QString::fromStdString(PVRush::PVNraw::default_tmp_path));
	if (!nraw_dir.exists()){
		nraw_dir.mkdir(QString::fromStdString(PVRush::PVNraw::default_tmp_path));
		delete_nraw_parent_dir = true;
	}

	env.compute_mapping();
	Inendi::PVView* view = env.compute_plotting()->current_view();

	// Export selection to temporary file
	Inendi::PVSelection sel(view->get_row_count());
	sel.select_all();

	std::string output_file = pvtest::get_tmp_filename();
	std::ofstream stream(output_file);

	PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const PVCore::PVColumnIndexes& col_indexes = view->get_axes_combination().get_original_axes_indexes();
	nraw.export_lines(stream, sel, col_indexes, 0, nraw.get_row_count());
	stream.flush();

	// Compare files content
	bool same_content = PVRush::PVUtils::files_have_same_content(argv[1], output_file);
	std::remove(output_file.c_str());

	// Remove nraw folder
	PVCore::PVDirectory::remove_rec(delete_nraw_parent_dir ? nraw_dir.path() : QString::fromStdString(nraw.collection().rootdir()));

	return (!same_content)*5;
}
