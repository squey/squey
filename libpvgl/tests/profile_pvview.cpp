
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVFileDescription.h>

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVView_types.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVPlotted.h>

#include <pvsdk/PVMessenger.h>

#include <pvgl/PVMain.h>
#include <pvgl/PVGLThread.h>

#include <QApplication>

#include <iostream>

#include "test-env.h"

#define IFILE   1
#define IFORMAT 2

PVSDK::PVMessenger* g_msg;

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
						foreach (Picviz::PVView_p update_view, views) {
							update_view->process_from_layer_stack();
							gl_update_view(update_view);
						}
						break;
					}
				default:
					break;
			};
		}
		else {
			usleep(100000);
		}
	}
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
	PVRush::PVControllerJob_p job = src->extract_from_agg_nlines(0, 30000000);
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
	// Create the PVSource objects
	Picviz::PVRoot_p root(new Picviz::PVRoot());

	Picviz::PVScene_p scene(new Picviz::PVScene("scene", root.get()));
	QList<Picviz::PVMapped_p> mappeds;
	QList<Picviz::PVPlotted_p> plotteds;
	QList<Picviz::PVView_p> views;

	// load the source
	PVLOG_INFO("loading file  : %s\n", argv[IFILE]);
	PVLOG_INFO("        format: %s\n", argv[IFORMAT]);
	PVLOG_INFO("step: %s - %d\n", __FILE__, __LINE__);
	Picviz::PVSource_p src = create_src (argv[IFILE], argv[IFORMAT]);
	PVLOG_INFO("step: %s - %d\n", __FILE__, __LINE__);
	Picviz::PVMapped_p mapped(new Picviz::PVMapped(src.get()));
	PVLOG_INFO("step: %s - %d\n", __FILE__, __LINE__);
	Picviz::PVPlotted_p plotted(new Picviz::PVPlotted(mapped.get()));
	//Picviz::PVPlotted_p plotted2(new Picviz::PVPlotted(Picviz::PVPlotting(mapped.get())));
	PVLOG_INFO("step: %s - %d\n", __FILE__, __LINE__);
	//mapped->add_plotted(plotted2);
	PVLOG_INFO("step: %s - %d\n", __FILE__, __LINE__);
	PVLOG_INFO("step: %s - %d\n", __FILE__, __LINE__);
	scene->add_child(src.get());
	PVLOG_INFO("step: %s - %d\n", __FILE__, __LINE__);
	views << plotted->get_view();
	PVLOG_INFO("step: %s - %d\n", __FILE__, __LINE__);

	PVGL::PVGLThread* th_pvgl = new PVGL::PVGLThread();
	g_msg = th_pvgl->get_messenger();
	th_pvgl->start();
	boost::thread th_main(boost::bind(thread_main, views));

	app.exec();

	th_main.join();
	th_pvgl->wait();

	return 0;
}
