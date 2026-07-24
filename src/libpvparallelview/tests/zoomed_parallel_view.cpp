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

#include <iostream>

#include <pvkernel/core/squey_bench.h>

#include <squey/PVAxis.h>
#include <squey/PVScaled.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVViewRenderingContext.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>

#include <pvbase/general.h>

#include "common.h"

#include <QApplication>
#include <QGraphicsView>

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

/*****************************************************************************/

int main(int argc, char** argv)
{
	if (argc < 2) {
		return 1;
	}

	QApplication app(argc, argv);

	PVParallelView::common::RAII_backend_init resources;

	TestEnv env(filename, fileformat);

	PVParallelView::PVViewRenderingContext* context = env.get_rendering_context();
	Squey::PVView& view = *context->lib_view();
	auto* zpview = new PVParallelView::PVZoomedParallelView(view.get_axes_combination());
	auto* scene = new PVParallelView::PVZoomedParallelScene(zpview, view, *context, PVCombCol(1));
	zpview->set_scene(scene);
	zpview->resize(1024, 1024);
	zpview->show();

	app.exec();

	return 0;
}
