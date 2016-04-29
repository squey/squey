/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <inendi/PVMapping.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotting.h>
#include <inendi/PVPlotted.h>
#include <cstdlib>
#include <iostream>
#include <QCoreApplication>
#include "test-env.h"

// FIXME: see PVLayer.cpp
#include <inendi/PVView.h>

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format [raw_dump] [raw_dump_transpose] [output]"
		          << std::endl;
		return 1;
	}

	init_env();
	PVCore::PVIntrinsics::init_cpuid();
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

	bool raw_dump = false;
	bool raw_dump_transp = false;
	QString out_path("plotted.out");
	if (argc >= 4) {
		raw_dump = argv[3][0] == '1';
		if (argc >= 5) {
			raw_dump_transp = argv[4][0] == '1';
			if (argc >= 6) {
				out_path = argv[5];
			}
		}
	}

	// Create the PVSource object
	Inendi::PVRoot_p root(new Inendi::PVRoot());
	Inendi::PVScene_p scene(new Inendi::PVScene("scene"));
	root->do_add_child(scene);
	Inendi::PVSource_sp src(
	    new Inendi::PVSource(PVRush::PVInputType::list_inputs() << file, sc_file, format));
	scene->add_source(src);
	Inendi::PVMapped_p mapped = src->emplace_add_child();
	PVRush::PVControllerJob_p job;

	if (raw_dump) {
		job = src->extract();
	} else {
		job = src->extract_from_agg_nlines(0, 200000000);
	}

	src->wait_extract_end(job);
	PVLOG_INFO("Extracted %u lines...\n", src->get_row_count());

	// Map the nraw
	mapped->process_from_parent_source();

	// And plot the mapped values
	Inendi::PVPlotted_p plotted(new Inendi::PVPlotted());
	mapped->do_add_child(plotted);
	plotted->set_plotting(Inendi::PVPlotting_p(new Inendi::PVPlotting(plotted.get())));
	plotted->process_from_parent_mapped();

	if (raw_dump) {
		PVLOG_INFO("Writing output...\n");
		plotted->dump_buffer_to_file(out_path, raw_dump_transp);
		PVLOG_INFO("Done !\n");
	} else {
		// Dump the mapped table to stdout in a CSV format
		// plotted->to_csv();
	}

	return 0;
}
