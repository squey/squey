/**
 * \file axes-comb.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

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
#include <pvkernel/core/PVSharedPointer.h>

#include <QApplication>

#include "test-env.h"

#include "axes-comb_dlg.h"

typedef PVCore::PVSharedPtr<Picviz::PVView> PVView_p;

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

class PVViewObs : public PVHive::PVObserver<Picviz::PVView>
{
public:
	PVViewObs(boost::thread & thread) : _thread(thread) {}

	void about_to_be_deleted()
	{
		std::cout << "Killing boost::thread" << std::endl;
		_thread.detach();
	}
private:
	boost::thread & _thread;
};


void thread(PVView_p& view_p)
{
	std::cout << "thread (" << boost::this_thread::get_id() << ")" << std::endl;
	std::cout << "Usage : <axis index> <axis name> | \"destroy\" + enter" << std::endl;

	while(true) {
		std::string cmd;
		getline(std::cin, cmd);

		if (QString(cmd.c_str()) == "destroy") {
			view_p.reset();
			break;
		}
		else {
			int axis_index;
			char axis_name[128];

			sscanf(cmd.c_str(), "%d %s", &axis_index, &axis_name);
			PVHive::PVActor<Picviz::PVView> actor;
			PVHive::PVHive::get().register_actor(view_p, actor);
			PVACTOR_CALL(actor, &Picviz::PVView::set_axis_name, axis_index, boost::cref(QString(axis_name)));
		}
	}
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

	PVView_p view_p = PVView_p(plotted->get_view().get());

	boost::thread th(boost::bind(thread, boost::ref(view_p)));

	PVViewObs view_observer = PVViewObs(th);
	PVHive::PVHive::get().register_observer(
		view_p,
		view_observer
	);

	TestDlg dlg(view_p);


	dlg.show();
	app.exec();
	th.join();

	return 0;
	// Segfault because PVView_p has already been force-deleted...
}
