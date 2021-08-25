//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/inendi_intrin.h>

#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>
#include <inendi/PVRoot.h>

#include <pvdisplays/PVDisplayIf.h>

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
	Inendi::PVRoot root;
	Inendi::PVSource& src = get_src_from_file(root, argv[1], argv[2]);
	src.emplace_add_child()   // Mapped
	    .emplace_add_child()  // Plotted
	    .emplace_add_child(); // View
	Inendi::PVView* view = src.current_view();

	QApplication app(argc, argv);

	PVParallelView::common::RAII_backend_init backend_resources;
	// Will also register displays
	PVGuiQt::common::register_displays();

	// Display all the possible Qt displays of this view and source
	PVDisplays::visit_displays_by_if<PVDisplays::PVDisplayViewIf>(
	    [&](PVDisplays::PVDisplayViewIf& obj) {
		    QWidget* w = PVDisplays::get_widget(obj, view);
		    w->show();
		});

	PVDisplays::visit_displays_by_if<PVDisplays::PVDisplaySourceIf>(
	    [&](PVDisplays::PVDisplaySourceIf& obj) {
		    QWidget* w = PVDisplays::get_widget(obj, &src);
		    w->show();
		});

	app.exec();

	return 0;
}
