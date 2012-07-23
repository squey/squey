/**
 * \file ad2g_widget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <picviz/widgets/PVAD2GWidget.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVView_types.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVAD2GView.h>
#include <pvsdk/PVMessenger.h>
#include <pvgl/PVMain.h>
#include <pvgl/PVGLThread.h>
#include <tulip/Node.h>
#include <tulip/Edge.h>
#include <cstdlib>
#include <iostream>
#include <QApplication>
#include "test-env.h"

PVSDK::PVMessenger* g_msg;

Picviz::PVAD2GView_p g_ad2gv;

void gl_update_view(Picviz::PVView_p view)
{
	PVSDK::PVMessage message;
	message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_VIEW;
	message.pv_view = view;
	message.int_1 = PVSDK_MESSENGER_REFRESH_SELECTION;
	g_msg->post_message_to_gl(message);

}

void thread_main(QList<Picviz::PVView_p> views)
{
	// Add views to PVGL
	foreach(Picviz::PVView_p const& view, views) {
		PVSDK::PVMessage msg;
		msg.function = PVSDK_MESSENGER_FUNCTION_PLEASE_WAIT;
		msg.pointer_1 = new QString(view->get_window_name());
		g_msg->post_message_to_gl(msg);

		msg.function = PVSDK_MESSENGER_FUNCTION_CREATE_VIEW;
		msg.pv_view = view;
		g_msg->post_message_to_gl(msg);
	}


	// Process selection messages
	PVSDK::PVMessage message;
	while (true) {
		if (g_msg->get_message_for_qt(message)) {
			Picviz::PVView_p view = message.pv_view;
			switch (message.function) {
				case PVSDK_MESSENGER_FUNCTION_SELECTION_CHANGED:
					{
						PVLOG_INFO("Selection changed %p. Launch graph..\n", view.get());
						g_ad2gv->pre_process();
						g_ad2gv->run(view.get());
						PVLOG_INFO("Graph done !\n");
						foreach (Picviz::PVView_p update_view, views) {
							gl_update_view(update_view);
						}
						break;
					}
				default:
					break;
			};
		}
		else {
			sleep(1);
		}
	}
}

Picviz::PVSource_p create_src(Picviz::PVScene_p scene, const QString &path_file, const QString &path_format);

int main(int argc, char** argv)
{
	if (argc <= 6) {
		std::cerr << "Usage: " << argv[0] << " file1 format1 file2 format2 file3 format3" << std::endl;
		return 1;
	}

	init_env();
	QApplication app(argc, argv);
	// Create the PVSource objects
	Picviz::PVRoot_p root;

	int argcount = 1;

	Picviz::PVScene_p scene(root, "scene");
	QList<Picviz::PVMapped_p> mappeds;
	QList<Picviz::PVPlotted_p> plotteds;
	QList<Picviz::PVView_p> views;

	g_ad2gv.reset(new Picviz::PVAD2GView(scene.get()));

	while (argcount < argc) {
		// load a source
		PVLOG_INFO("loading file  : %s\n", argv[argcount]);
		PVLOG_INFO("        format: %s\n", argv[argcount+1]);
		Picviz::PVSource_p src = create_src(scene, argv[argcount], argv[argcount+1]);
		Picviz::PVMapped_p mapped(src);
		Picviz::PVPlotted_p plotted(mapped);
		mapped->process_from_parent_source(false);
		plotted->process_from_parent_mapped(false);
		//Picviz::PVPlotted_p plotted2(new Picviz::PVPlotted(Picviz::PVPlotting(mapped.get())));
		//mapped->add_plotted(plotted2);
		scene->add_child(src);
		views << plotted->get_view();

		// create the corresponding node
		//g_ad2gv->add_view(plotted->get_view().get());

		// next!
		argcount += 2;

	}

	PVLOG_INFO("all loaded\n");

	QMainWindow *mw = new QMainWindow();
	PVWidgets::PVAD2GWidget* ad2g_widget = new PVWidgets::PVAD2GWidget(g_ad2gv, mw);
	mw->setCentralWidget(ad2g_widget);
	mw->show();

	PVGL::PVGLThread* th_pvgl = new PVGL::PVGLThread();
	g_msg = th_pvgl->get_messenger();
	th_pvgl->start();
	boost::thread th_main(boost::bind(thread_main, views));

	app.exec();

	th_main.join();
	th_pvgl->wait();

	return 0;
}

Picviz::PVSource_p create_src(Picviz::PVScene_p scene, const QString &path_file, const QString &path_format)
{
	// Input file
	PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

	// Load the given format file
	PVRush::PVFormat format("format", path_format);
	if (!format.populate()) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return Picviz::PVSource_p::invalid();
	}

	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return Picviz::PVSource_p::invalid();
	}

	Picviz::PVSource_p src(scene, PVRush::PVInputType::list_inputs() << file, sc_file, format);
	src->get_extractor().get_agg().set_strict_mode(true);
	PVRush::PVControllerJob_p job = src->extract_from_agg_nlines(0, 200000);
	job->wait_end();

	return src;
}
