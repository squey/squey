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

#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>
#include <inendi/PVRoot.h>

#include <pvguiqt/PVLayerStackDelegate.h>
#include <pvguiqt/PVLayerStackModel.h>
#include <pvguiqt/PVLayerStackView.h>

#include <QApplication>
#include <QMainWindow>
#include <QTableView>
#include <QVBoxLayout>

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

	// Qt app
	QApplication app(argc, argv);

	Inendi::PVView& view = *src.current_view();
	PVGuiQt::PVLayerStackDelegate* delegate = new PVGuiQt::PVLayerStackDelegate(view);
	PVGuiQt::PVLayerStackModel* model = new PVGuiQt::PVLayerStackModel(view);
	PVGuiQt::PVLayerStackModel* model2 = new PVGuiQt::PVLayerStackModel(view);

	PVGuiQt::PVLayerStackView* qt_view = new PVGuiQt::PVLayerStackView();
	PVGuiQt::PVLayerStackView* qt_view2 = new PVGuiQt::PVLayerStackView();
	qt_view->setModel(model);
	qt_view->setItemDelegate(delegate);
	qt_view2->setModel(model2);
	qt_view2->setItemDelegate(delegate);

	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(qt_view);

	QMainWindow* mw2 = new QMainWindow();
	mw2->setCentralWidget(qt_view2);

	mw->show();
	mw2->show();

	int ret = app.exec();

	return ret;
}
