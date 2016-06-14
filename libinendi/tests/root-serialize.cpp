/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>
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
	Inendi::PVRoot root;
	Inendi::PVScene& scene = root.emplace_add_child("scene");
	scene.emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
	Inendi::PVSource& src =
	    scene.emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
	PVRush::PVControllerJob_p job = src.extract();
	job->wait_end();
	auto& plotted = src.emplace_add_child().emplace_add_child();
	plotted.emplace_add_child();
	plotted.emplace_add_child();

	Inendi::PVScene& scene2 = root.emplace_add_child("scene1");
	Inendi::PVSource& src2 =
	    scene2.emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
	auto& plotted2 = src2.emplace_add_child().emplace_add_child();
	plotted2.emplace_add_child();
	plotted2.emplace_add_child();

	// Serialize the root object
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchive(
	    "/srv/tmp-inendi/test", PVCore::PVSerializeArchive::write, 1));
	ar->get_root()->object("root", root);
	ar->finish();

	Inendi::PVRoot root_read;

	// Get it back !
	ar.reset(new PVCore::PVSerializeArchive("/srv/tmp-inendi/test",
	                                        PVCore::PVSerializeArchive::read, 1));
	ar->get_root()->object("root", root_read);
	ar->finish();

	return 0;
}
