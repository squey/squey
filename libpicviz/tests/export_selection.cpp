/**
 * \file export_selection.cpp
 *
 * Copyright (C) Picviz Labs 2011-2014
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
#include <picviz/PVSelection.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>
#include <cstdlib>
#include <iostream>
#include <QCoreApplication>
#include <QFile>
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	init_env();
	PVCore::PVIntrinsics::init_cpuid();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	bool delete_nraw_parent_dir = false;
	QDir nraw_dir(PVRush::PVNraw::default_tmp_path);
	if (!nraw_dir.exists()){
		nraw_dir.mkdir(PVRush::PVNraw::default_tmp_path);
		delete_nraw_parent_dir = true;
	}

	// Input file
	QString path_file(argv[1]);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	QString path_format(argv[2]);
	PVRush::PVFormat format("format", path_format);
	if (!format.populate()) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return 2;
	}

	// Get the source creator
	QString file_path(argv[1]);
	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return 3;
	}

	// Create the PVSource object
	Picviz::PVRoot_p root;
	Picviz::PVScene_p scene(root, "scene");
	Picviz::PVSource_p src(scene, PVRush::PVInputType::list_inputs() << file, sc_file, format);
	PVRush::PVControllerJob_p job = src->extract();
	job->wait_end();
	PVLOG_INFO("Extracted %u lines...\n", src->get_row_count());

	// Map the nraw
	Picviz::PVMapped_p mapped(src);
	mapped->process_from_parent_source();

	// And plot the mapped values
	Picviz::PVPlotted_p plotted(mapped);
	plotted->process_from_parent_mapped();
	Picviz::PVView* view = src->current_view();

	// Export selection to temporary file
	Picviz::PVSelection& sel = view->get_real_output_selection();
	sel.select_all();
	QTemporaryFile output_file;
	QTextStream stream(&output_file);
	if (!output_file.open()) {
		return 4;
	}
	PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const PVCore::PVColumnIndexes& col_indexes = view->get_axes_combination().get_original_axes_indexes();
	nraw.export_lines(stream, sel, col_indexes, 0, src->get_row_count());
	stream.flush();

	// Compare files content
	bool same_content = PVRush::PVUtils::files_have_same_content(argv[1], output_file.fileName());

	// Remove nraw folder
	PVCore::PVDirectory::remove_rec(delete_nraw_parent_dir ? nraw_dir.path() : QString(nraw.get_nraw_folder().c_str()));

	return (!same_content)*5;
}
