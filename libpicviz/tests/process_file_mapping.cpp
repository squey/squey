/**
 * \file process_file_mapping.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/picviz_intrin.h>
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
#include <cstdlib>
#include <iostream>
#include <QCoreApplication>
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
	Picviz::PVRoot_p root;
	Picviz::PVScene_p scene(root, "scene");
	Picviz::PVSource_p src(scene, PVRush::PVInputType::list_inputs() << file, sc_file, format);
	Picviz::PVMapped_p mapped(src);
	//src->set_invalid_elts_mode(true);
	PVRush::PVControllerJob_p job = src->extract();
	src->wait_extract_end(job);
	PVLOG_INFO("Extraction job bytes: %lu, time: %0.4fs, mean bw: %0.4f MB/s\n", job->total_bytes_processed(), job->duration().seconds(), job->mean_bw());

	mapped->to_csv();
	//
	/*QStringList const& inv(job->get_invalid_elts());
	foreach (QString const& sinv, inv) {
		PVLOG_INFO("invalid: %s\n", qPrintable(sinv));
	}*/

	// Map the nraw
	//mapped->process_from_parent_source();
	// Dump the mapped table to stdout in a CSV format
	//mapped->to_csv();

	// Save current mapped table
#if 0
	Picviz::PVMapped_p mapped(src);
	Picviz::PVMapped::mapped_table_t save(mapped->get_table());

	mapped->invalidate_all();
	mapped->process_from_parent_source();
	//mapped->to_csv();
	
	// Compare table
	ASSERT_VALID(save.size() == mapped->get_table().size());
	for (size_t i = 0; i < save.size(); i++) {
		Picviz::PVMapped::mapped_row_t const& rsave = save[i];
		Picviz::PVMapped::mapped_row_t const& rcmp  = mapped->get_table()[i];

		ASSERT_VALID(rsave.size() == rcmp.size());
		//ASSERT_VALID(memcmp(&rsave.at(0), &rcmp.at(0), rsave.size()*sizeof(Picviz::PVMapped::decimal_storage_type)) == 0);
		write(4, &rsave.at(0), rsave.size()*sizeof(Picviz::PVMapped::decimal_storage_type));
		write(5, &rcmp.at(0),  rcmp.size()*sizeof(Picviz::PVMapped::decimal_storage_type));
	}
#endif

	return 0;
}
