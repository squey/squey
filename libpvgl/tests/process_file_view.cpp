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
#include <picviz/PVPlotting.h>
#include <picviz/PVPlotted.h>
#include <pvsdk/PVMessenger.h>
#include <pvgl/PVMain.h>
#include <pvgl/PVGLThread.h>
#include <cstdlib>
#include <iostream>
#include <QCoreApplication>
#include "test-env.h"

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
		msg.pointer_1 = new QString("test");
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
						PVLOG_INFO("Selection changed %p.\n", view.get());
						Picviz::PVSelection const& sel = view->get_real_output_selection();
						foreach (Picviz::PVView_p change_view, views) {
							if (change_view != view) {
								PVLOG_INFO("Update view %p..\n", change_view.get());
								change_view->set_selection_view(sel);
								gl_update_view(change_view);
							}
						}
						break;
					}
				default:
					break;
			};
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
	Picviz::PVRoot_p root(new Picviz::PVRoot());
	Picviz::PVSource_p src(new Picviz::PVSource(PVRush::PVInputType::list_inputs() << file, sc_file, format));
	PVRush::PVControllerJob_p job = src->extract();
	job->wait_end();

	// Map the nraw
	Picviz::PVMapped_p mapped(new Picviz::PVMapped(Picviz::PVMapping(src.get())));

	// And plot the mapped values
	Picviz::PVPlotted_p plotted(new Picviz::PVPlotted(Picviz::PVPlotting(mapped.get())));
	Picviz::PVPlotted_p plotted2(new Picviz::PVPlotted(Picviz::PVPlotting(mapped.get())));

	QList<Picviz::PVView_p> views;
	views << plotted->get_view();
	views << plotted2->get_view();

	PVGL::PVGLThread* th_pvgl = new PVGL::PVGLThread();
	g_msg = th_pvgl->get_messenger();
	th_pvgl->start();
	boost::thread th_main(boost::bind(thread_main, views));

	th_main.join();
	th_pvgl->wait();

	return 0;
}
