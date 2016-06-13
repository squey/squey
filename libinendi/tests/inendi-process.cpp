/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_bench.h>
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
#include <inendi/PVView.h>

#include "test-env.h"

int main(int argc, char** argv)
{
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
	Inendi::PVRoot_p root(new Inendi::PVRoot());
	Inendi::PVScene& scene = root->emplace_add_child("scene");
	Inendi::PVSource& src =
	    scene.emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
	PVRush::PVControllerJob_p job = src.extract();
	job->wait_end();
	PVLOG_INFO("Extracted %u lines...\n", src.get_row_count());

	// Map the nraw
	Inendi::PVMapped& mapped = src.emplace_add_child();

	// And plot the mapped values
	Inendi::PVPlotted& plotted = mapped.emplace_add_child();
	plotted.emplace_add_child();
	Inendi::PVView* view = src.current_view();

	BENCH_START(ls);
	view->process_layer_stack();
	BENCH_END(ls, "layer-stack-1", 1, 1, 1, 1);

	for (int i = 0; i < 9; i++) {
		view->add_new_layer();
	}

	BENCH_START(ls2);
	view->process_layer_stack();
	BENCH_END(ls2, "layer-stack-10", 1, 1, 1, 1);

	BENCH_START(sel);
	view->process_selection();
	BENCH_END(sel, "selection", 1, 1, 1, 1);

	BENCH_START(el);
	view->process_eventline();
	BENCH_END(el, "eventline", 1, 1, 1, 1);

	BENCH_START(visibility);
	view->process_visibility();
	BENCH_END(visibility, "visibility", 1, 1, 1, 1);

	return 0;
}
