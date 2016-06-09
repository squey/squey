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
	Inendi::PVRoot_p root(new Inendi::PVRoot());
	Inendi::PVScene& scene = root->emplace_add_child("scene");
	Inendi::PVSource& src =
	    scene.emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
	PVRush::PVControllerJob_p job = src.extract();
	job->wait_end();

	// Dump the NRAW
	src.get_rushnraw().dump_csv();

	// Serialize the scene
	PVCore::PVSerializeArchive_p ar(
	    new PVCore::PVSerializeArchive("/tmp/test", PVCore::PVSerializeArchive::write, 1));
	ar->get_root()->object("scene", scene);
	ar->finish();

	// Get it back !
	Inendi::PVScene& scene2 = root->emplace_add_child("scene");
	ar.reset(new PVCore::PVSerializeArchive("/tmp/test", PVCore::PVSerializeArchive::read, 1));
	ar->get_root()->object("scene", scene2);
	ar->finish();

	auto new_src = scene2.get_children().front();

	job = new_src->extract();
	job->wait_end();

	std::cerr << "--------" << std::endl
	          << "New output: " << std::endl
	          << "----------" << std::endl
	          << std::endl;
	// Dump the NRAW
	new_src->get_rushnraw().dump_csv();

	return 0;
}
