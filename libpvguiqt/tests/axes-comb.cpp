/**
 * \file axes-comb.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSharedPointer.h>
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

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVAxesListModel.h>

#include <QApplication>
#include <QListView>
#include <QMainWindow>

#include "test-env.h"

#include "axes-comb_dlg.h"

Picviz::PVSource_p create_src(const QString &path_file, const QString &path_format)
{
	Picviz::PVRoot_p root;
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

	Picviz::PVScene_p scene(root);
        Picviz::PVSource_p src(scene, PVRush::PVInputType::list_inputs() << file, sc_file, format);
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


void thread(Picviz::PVView* view_p)
{
	std::cout << "thread (" << boost::this_thread::get_id() << ")" << std::endl;
	std::cout << "Usage : <axis index> <axis name> | \"destroy\" + enter" << std::endl;

	PVHive::PVActor<Picviz::PVView> actor;
	{
		Picviz::PVView_sp v_sp = view_p->shared_from_this();
		PVHive::PVHive::get().register_actor(v_sp, actor);
	}

	while(true) {
		std::string cmd;
		getline(std::cin, cmd);

		if (QString(cmd.c_str()) == "destroy") {
			view_p->remove_from_tree();
			break;
		}
		else {
			int axis_index;
			char axis_name[128];

			sscanf(cmd.c_str(), "%d %s", &axis_index, (char*) &axis_name);
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

	PVCore::PVIntrinsics::init_cpuid();

	init_env();
	QApplication app(argc, argv);

	PVLOG_INFO("loading file  : %s\n", argv[1]);
	PVLOG_INFO("        format: %s\n", argv[2]);

	Picviz::PVSource_p src = create_src (argv[1], argv[2]);
	Picviz::PVMapped_p mapped(src);
	Picviz::PVPlotted_p plotted(mapped);

	Picviz::PVView_p view_p(plotted);

	boost::thread th(boost::bind(thread, view_p.get()));
	PVViewObs view_observer = PVViewObs(th);
	PVHive::PVHive::get().register_observer(
			view_p,
			view_observer
			);

	TestDlg *dlg = new TestDlg(view_p);
	dlg->show();

	PVGuiQt::PVAxesListModel* model = new PVGuiQt::PVAxesListModel(view_p);
	QListView* view = new QListView();
	view->setModel(model);
	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(view);
	mw->show();

	PVGuiQt::PVAxesCombinationDialog* axes_dlg = new PVGuiQt::PVAxesCombinationDialog(view_p);
	axes_dlg->show();

	view_p.reset();


	app.exec();
	th.join();

	return 0;
	// Segfault because PVView_p has already been force-deleted...
}
