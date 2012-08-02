/**
 * \file process_file_plotting.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVPlotted.h>
#include <cstdlib>
#include <iostream>
#include <QCoreApplication>
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format [raw_dump] [raw_dump_transpose]" << std::endl;
		return 1;
	}

#if 0
	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	QString path_format(argv[2]);
	PVRush::PVFormat format("format", path_format);
	if (!format.populate()) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return false;
	}

	// Get the source creator
	QString file_path(argv[1]);
	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return 1;
	}

	// Create the PVSource object
	Picviz::PVRoot_p root(new Picviz::PVRoot());
	Picviz::PVSource_p src(new Picviz::PVSource(PVRush::PVInputType::list_inputs() << file, sc_file, format));
	PVRush::PVControllerJob_p job = src->extract_from_agg_nlines(0, 40000000);
	job->wait_end();
	PVLOG_INFO("Extracted %u lines...\n", src->get_row_count());
#endif
	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	QString path_format(argv[2]);
	PVRush::PVFormat format("format", path_format);
	if (!format.populate()) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return false;
	}

	// Get the source creator
	QString file_path(argv[1]);
	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return 1;
	}

	// Create the PVSource object
	Picviz::PVRoot_p root(new Picviz::PVRoot());
	Picviz::PVSource_p src(new Picviz::PVSource(PVRush::PVInputType::list_inputs() << file, sc_file, format));
	PVRush::PVControllerJob_p job = src->extract();
	job->wait_end();
	PVLOG_INFO("Extracted %u lines...\n", src->get_row_count());

	// Map the nraw
	Picviz::PVMapped_p mapped(new Picviz::PVMapped(Picviz::PVMapping(src.get())));

	// And plot the mapped values
	Picviz::PVPlotted_p plotted(new Picviz::PVPlotted(Picviz::PVPlotting(mapped.get())));

	bool raw_dump = false;
	bool raw_dump_transp = false;
	if (argc >= 4) {
		raw_dump = argv[3][0] == '1';
		if (argc >= 5) {
			raw_dump_transp = argv[4][0] == '1';
		}
	}

	PVLOG_INFO("Writing output...\n");
	if (raw_dump) {
		plotted->dump_buffer_to_file("plotted.out", raw_dump_transp);
	}
	else {
		// Dump the mapped table to stdout in a CSV format
		plotted->to_csv();
	}
	PVLOG_INFO("Done !\n");

	return 0;
}
