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
	scene->add_source(src);
	PVRush::PVControllerJob_p job = src->extract();
	job->wait_end();

	// Dump the NRAW
	src->get_rushnraw().dump_csv();

	// Serialize the scene
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchive("/tmp/test", PVCore::PVSerializeArchive::write, 1));
	ar->get_root()->object("scene", *scene);
	ar->finish();

	// Get it back !
	src.reset();
	scene = Picviz::PVScene_p(root, "scene");
	ar.reset(new PVCore::PVSerializeArchive("/tmp/test", PVCore::PVSerializeArchive::read, 1));
	ar->get_root()->object("scene", *scene);
	ar->finish();

	Picviz::PVScene::list_sources_t srcs = scene->get_sources(*sc_file->supported_type_lib());
	if (srcs.size() != 1) {
		std::cerr << "No source was recreated !" << std::endl;
		return 1;
	}
	src = srcs.at(0);
	
	job = src->extract();
	job->wait_end();

	std::cerr << "--------" << std::endl << "New output: " << std::endl << "----------" << std::endl << std::endl;
	// Dump the NRAW
	src->get_rushnraw().dump_csv();

	return 0;
}
