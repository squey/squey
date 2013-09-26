/**
 * \file nraw_count_by.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */
/**
 * \file nraw_create.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#define SIMULATE_PIPELINE
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/picviz_bench.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"
#include <QCoreApplication>
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " input_log_file format [lines_count=1000000]" << std::endl;
		return 1;
	}
	uint64_t nb_lines = 1000000;
	if (argc >= 4) {
		nb_lines = atoi(argv[3]);
	}

	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	QString path_format(argv[2]);
	PVLOG_INFO("Load format...\n");
	PVRush::PVFormat format("format", path_format);
	if (!format.populate(true)) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return 1;
	}
	PVLOG_INFO("Format loaded.\n");

	// Get the source creator
	QString file_path(argv[1]);
	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return 1;
	}

	// Process that file with the found source creator thanks to the extractor
	PVLOG_INFO("Creating source...\n");
	PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file, format);
	if (!src) {
		std::cerr << "Unable to create PVRush source from file " << argv[1] << std::endl;
		return 1;
	}
	PVLOG_INFO("Source created.\n");

	// Create the extractor
	PVRush::PVExtractor ext;
	ext.start_controller();
	ext.add_source(src);
	ext.set_format(format);
	ext.set_chunk_filter(format.create_tbb_filters());

	PVLOG_INFO("Asking for %lu lines...\n", nb_lines);
	PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(0, nb_lines);
	job->wait_end();

	PVCore::PVSelBitField sel;
	sel.select_all();
	PVRush::PVNraw::count_by_t count_by;

	PVLOG_INFO("count_by_with_sel:\n");
	ext.get_nraw().count_by_with_sel(3, 6, count_by, sel);
	/*for (auto& v1 : count_by) {
		std::cout << v1.first << ":" << std::endl;
		for (auto& v2 : count_by[v1.first]) {
			std::cout << "   " << v2.first << ": " << v2.second << std::endl;
		}
		std::cout << "-----------------------" << std::endl;
	}*/

	return 0;
}
