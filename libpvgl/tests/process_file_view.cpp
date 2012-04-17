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

Picviz::PVAD2GView *g_ad2gv;

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
						/*foreach (Picviz::PVView_p change_view, views) {
							if (change_view != view) {
								PVLOG_INFO("Update view %p..\n", change_view.get());
								Picviz::PVSelection new_sel = (*cf_p)(*view, *change_view);
								PVLOG_INFO("Selection computed for %p..\n", change_view.get());
								change_view->set_selection_view(new_sel);
								gl_update_view(change_view);
							}
						}*/
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
	}
}

Picviz::PVSource_p create_src(const QString &path_file, const QString &path_format);

int main(int argc, char** argv)
{
	if (argc <= 6) {
		std::cerr << "Usage: " << argv[0] << " file1 format1 file2 format2 file3 format3" << std::endl;
		return 1;
	}

	init_env();
	QApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();
	// Create the PVSource objects
	Picviz::PVRoot_p root(new Picviz::PVRoot());

	int argcount = 1;

	Picviz::PVScene_p scene(new Picviz::PVScene("scene", root.get()));
	QList<Picviz::PVSource_p> srcs;
	QList<Picviz::PVMapped_p> mappeds;
	QList<Picviz::PVPlotted_p> plotteds;
	QList<Picviz::PVView_p> views;

	g_ad2gv = new Picviz::PVAD2GView(scene.get());

	while (argcount < argc) {
		// load a source
		PVLOG_INFO("loading file  : %s\n", argv[argcount]);
		PVLOG_INFO("        format: %s\n", argv[argcount+1]);
		Picviz::PVSource_p src = create_src (argv[argcount], argv[argcount+1]);
		Picviz::PVMapped_p mapped(new Picviz::PVMapped(Picviz::PVMapping(src.get())));
		Picviz::PVPlotted_p plotted(new Picviz::PVPlotted(Picviz::PVPlotting(mapped.get())));
		srcs << src;
		mappeds << mapped;
		plotteds << plotted;
		views << plotted->get_view();

		// create the corresponding node
		g_ad2gv->add_view(plotted->get_view().get());

		// next!
		argcount += 2;

	}

	PVLOG_INFO("all loaded\n");



	Picviz::PVCombiningFunctionView* cf_p(new Picviz::PVCombiningFunctionView());
	Picviz::PVTFViewRowFiltering* tf = cf_p->get_first_tf();
	//Picviz::PVRFFAxesBind* rff_bind = new Picviz::PVRFFAxesBind();
	LIB_CLASS(Picviz::PVSelRowFilteringFunction) &row_filters = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get();
	Picviz::PVSelRowFilteringFunction_p rff_bind = row_filters.get_class_by_name("axes_bind");
	assert(rff_bind);
	rff_bind = rff_bind->clone<Picviz::PVSelRowFilteringFunction>();
	PVCore::PVArgumentList args;
	args["axis_org"].setValue(PVCore::PVAxisIndexType(1));
	args["axis_dst"].setValue(PVCore::PVAxisIndexType(1));
	rff_bind->set_args(args);
	//tf->push_rff(Picviz::PVSelRowFilteringFunction_p(rff_bind));
	tf->push_rff(rff_bind);


	// Create edges
	Picviz::PVCombiningFunctionView_p cf_sp(cf_p);
	g_ad2gv->set_edge_f(views[0].get(), views[1].get(), cf_sp);
	g_ad2gv->set_edge_f(views[1].get(), views[0].get(), cf_sp);
	g_ad2gv->set_edge_f(views[0].get(), views[2].get(), cf_sp);
	g_ad2gv->set_edge_f(views[2].get(), views[0].get(), cf_sp);
	g_ad2gv->set_edge_f(views[1].get(), views[2].get(), cf_sp);
	g_ad2gv->set_edge_f(views[2].get(), views[1].get(), cf_sp);

	/*QMainWindow *mw = new QMainWindow();
	Picviz::PVAD2GWidget* ad2g_widget = new Picviz::PVAD2GWidget(*g_ad2gv, mw);
	mw->setCentralWidget(ad2g_widget->get_widget());
	mw->show();*/

	PVGL::PVGLThread* th_pvgl = new PVGL::PVGLThread();
	g_msg = th_pvgl->get_messenger();
	th_pvgl->start();
	boost::thread th_main(boost::bind(thread_main, views));


	app.exec();

	th_main.join();
	th_pvgl->wait();

	return 0;
}

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
