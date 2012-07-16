
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVTests.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>

#include <QApplication>

#include "test-env.h"

#include "axes-comb_dlg.h"

Picviz::PVSource_p create_src(const QString &path_file, const QString &path_format)
{
        // Input file
        PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

        // Load the given format file
        PVRush::PVFormat format("format", path_format);
        if (!format.populate()) {
                std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
                return Picviz::PVSource_p();
        }

        PVRush::PVSourceCreator_p sc_file;
        if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
                return Picviz::PVSource_p();
        }

        Picviz::PVSource_p src(new Picviz::PVSource(PVRush::PVInputType::list_inputs() << file, sc_file, format));
        src->get_extractor().get_agg().set_strict_mode(true);
        PVRush::PVControllerJob_p job = src->extract_from_agg_nlines(0, 200000);
        job->wait_end();

        return src;
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	init_env();
	QApplication app(argc, argv);

	Picviz::PVRoot_p root(new Picviz::PVRoot());
	Picviz::PVScene_p scene(new Picviz::PVScene("scene", root.get()));
	QList<Picviz::PVMapped_p> mappeds;
	QList<Picviz::PVPlotted_p> plotteds;
	QList<Picviz::PVView_p> views;

	PVLOG_INFO("loading file  : %s\n", argv[1]);
	PVLOG_INFO("        format: %s\n", argv[2]);

	Picviz::PVSource_p src = create_src (argv[1], argv[2]);
	Picviz::PVMapped_p mapped(new Picviz::PVMapped(Picviz::PVMapping(src.get())));
	Picviz::PVPlotted_p plotted(new Picviz::PVPlotted(Picviz::PVPlotting(mapped.get())));

	mapped->add_plotted(plotted);
	src->add_mapped(mapped);
	scene->add_source(src);

	TestDlg dlg(plotted->get_view().get());
	dlg.show();

	app.exec();

	return 0;
}
