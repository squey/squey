#include <pvfilter/PVPluginsLoad.h>
#include <pvrush/PVInputType.h>
#include <pvrush/PVPluginsLoad.h>
#include <pvrush/PVExtractor.h>
#include <pvrush/PVControllerJob.h>
#include <pvrush/PVFormat.h>
#include <pvrush/PVTests.h>
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
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	init_env();
	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVCore::PVArgument file(path_file);

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
	Picviz::PVScene_p scene(new Picviz::PVScene("scene", root));
	Picviz::PVSource_p src(new Picviz::PVSource(scene));
	PVRush::PVControllerJob_p job = src->files_append(format, sc_file, PVRush::PVInputType::list_inputs() << file);
	job->wait_end();

	// Map the nraw
	Picviz::PVMapping_p mapping(new Picviz::PVMapping(src));
	Picviz::PVMapped_p mapped(new Picviz::PVMapped(mapping));

	// And plot the mapped values
	Picviz::PVPlotting_p plotting(new Picviz::PVPlotting(mapped));
	Picviz::PVPlotted_p plotted(new Picviz::PVPlotted(plotting));

	// Dump the mapped table to stdout in a CSV format
	plotted->to_csv();

	return 0;
}
