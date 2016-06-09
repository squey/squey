/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_assert.h>
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
#include <cstdlib>
#include <iostream>
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format [raw_dump]" << std::endl;
		return 1;
	}

	init_env();
	PVCore::PVIntrinsics::init_cpuid();
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

	bool raw_dump = true;
	if (argc >= 4) {
		raw_dump = argv[3][0] == '1';
	}

	// Create the PVSource object
	Inendi::PVRoot_p root(new Inendi::PVRoot());
	Inendi::PVScene& scene = root->emplace_add_child("scene");
	Inendi::PVSource& src =
	    scene.emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
	Inendi::PVMapped& mapped = src.emplace_add_child();
	PVRush::PVControllerJob_p job;

	if (raw_dump) {
		job = src.extract();
	} else {
		job = src.extract(0, 200000000);
	}

	src.wait_extract_end(job);

	if (raw_dump) {
		mapped.to_csv();
	} else {
		PVLOG_INFO("Extracted %u lines...\n", src.get_row_count());
	}

	return 0;
}
