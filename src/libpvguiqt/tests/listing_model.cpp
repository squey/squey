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

#include <pvkernel/core/squey_intrin.h>

#include <squey/PVMapped.h>
#include <squey/PVScaled.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>
#include <squey/PVRoot.h>

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingView.h>

#include <QApplication>
#include <QMainWindow>
#include <QTableView>
#include <QVBoxLayout>

#include <boost/thread.hpp>

#include "common.h"
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}
	init_env();

	// Get a SQUEY tree from the given file/format
	Squey::PVRoot root;
	Squey::PVSource& src = get_src_from_file(root, argv[1], argv[2]);
	src.emplace_add_child()   // Mapped
	    .emplace_add_child()  // Scaled
	    .emplace_add_child(); // View

	// Qt app
	QApplication app(argc, argv);

	Squey::PVView& view = *src.current_view();
	auto* model = new PVGuiQt::PVListingModel(view);

	auto* qt_view = new PVGuiQt::PVListingView(view);
	qt_view->setModel(model);

	auto* mw = new QMainWindow();
	mw->setCentralWidget(qt_view);

	mw->show();

	// Remove listing when pressing enter
	boost::thread key_thread([&] {
		std::cerr << "Press enter to remove data-tree..." << std::endl;
		while (getchar() != '\n')
			;
	});

	int ret = app.exec();
	key_thread.join();

	return ret;
}
