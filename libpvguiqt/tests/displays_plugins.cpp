/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/inendi_intrin.h>

#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <pvdisplays/PVDisplaysImpl.h>

#include <pvguiqt/common.h>

#include <pvparallelview/PVParallelView.h>

#include <QApplication>

#include <iostream>

#include "common.h"
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}
	PVCore::PVIntrinsics::init_cpuid();
	init_env();

	// Get a INENDI tree from the given file/format
	Inendi::PVRoot_p root;
	Inendi::PVSource_sp src = get_src_from_file(root, argv[1], argv[2]);
	src->create_default_view();
	Inendi::PVView* view = src->current_view();
	init_random_colors(*view);

	QApplication app(argc, argv);

	PVParallelView::common::RAII_cuda_init cuda_resources; // Will also register displays
	PVGuiQt::common::register_displays();

	// Display all the possible Qt displays of this view and source
	PVDisplays::get().visit_displays_by_if<PVDisplays::PVDisplayViewIf>(
		[&](PVDisplays::PVDisplayViewIf& obj)
		{
			QWidget* w = PVDisplays::get().get_widget(obj, view);
			w->show();
		});

	PVDisplays::get().visit_displays_by_if<PVDisplays::PVDisplaySourceIf>(
		[&](PVDisplays::PVDisplaySourceIf& obj)
		{
			QWidget* w = PVDisplays::get().get_widget(obj, src.get());
			w->show();
		});

	PVDisplays::get().visit_displays_by_if<PVDisplays::PVDisplayViewAxisIf>(
		[&](PVDisplays::PVDisplayViewAxisIf& obj)
		{
			QWidget* w = PVDisplays::get().get_widget(obj, view, 1);
			w->show();
		});

	app.exec();

	return 0;
}
