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
	Inendi::PVScene_p scene(new Inendi::PVScene("scene0"));
	root->do_add_child(scene);
	Inendi::PVSource_sp src(
	    new Inendi::PVSource(PVRush::PVInputType::list_inputs() << file, sc_file, format));
	scene->add_source(src);
	scene->add_source(src);
	PVRush::PVControllerJob_p job = src->extract();
	job->wait_end();
	src->create_default_view();
	src->create_default_view();

	Inendi::PVView& v0 = *src->get_children<Inendi::PVView>().at(0);
	Inendi::PVView& v1 = *src->get_children<Inendi::PVView>().at(1);

	Inendi::PVScene_p scene2(new Inendi::PVScene("scene1"));
	root->do_add_child(scene2);
	Inendi::PVSource_sp src2(
	    new Inendi::PVSource(PVRush::PVInputType::list_inputs() << file, sc_file, format));
	scene2->add_source(src2);
	src2->create_default_view();
	src2->create_default_view();

	Inendi::PVView& v2 = *src2->get_children<Inendi::PVView>().at(0);
	Inendi::PVView& v3 = *src2->get_children<Inendi::PVView>().at(1);

	// Serialize the root object
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchive(
	    "/srv/tmp-inendi/test", PVCore::PVSerializeArchive::write, 1));
	ar->get_root()->object("root", *root);
	ar->finish();

	src.reset();
	scene.reset();
	v0.remove_from_tree();
	v1.remove_from_tree();
	root.reset(new Inendi::PVRoot());

	// Get it back !
	ar.reset(new PVCore::PVSerializeArchive("/srv/tmp-inendi/test",
	                                        PVCore::PVSerializeArchive::read, 1));
	ar->get_root()->object("root", *root);
	ar->finish();

	root->dump();

#if 0
	Inendi::PVScene::list_sources_t srcs = scene->get_sources(*sc_file->supported_type_lib());
	if (srcs.size() != 1) {
		std::cerr << "No source was recreated !" << std::endl;
		return 1;
	}
	src = srcs.at(0)->shared_from_this();
	
	job = src->extract();
	job->wait_end();

	std::cerr << "--------" << std::endl << "New output: " << std::endl << "----------" << std::endl << std::endl;
	// Dump the NRAW
	src->get_rushnraw().dump_csv();
#endif

	return 0;
}
