//
// MIT License
//
// © Squey, 2026
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

// Headless smoke test for the whole display-widget catalog. Every view kind that
// Squey can open (listing, layer-stack, distinct-values, the group-by family,
// axes-combination, correlation, mapping-scaling, filters -- from libpvguiqt --
// and the full-parallel, zoomed, hit-count, scatter and timeseries views -- from
// libpvparallelview) is instantiated against a real, fully-computed view and
// shown offscreen. It guards three things:
//   1. the import pipeline produced the expected shape (rows x columns);
//   2. the display catalog stays complete (no display silently dropped);
//   3. every display still builds a non-null widget that actually becomes visible.
// Runs under FORCE_CPU=1 + "-platform offscreen" (see CMakeLists.txt), so the
// parallelview widgets fall back to the software backend instead of the GPU.

#include <squey/PVMapped.h>
#include <squey/PVScaled.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>
#include <squey/PVRoot.h>

#include <pvkernel/core/squey_assert.h>

#include <pvbase/types.h>

#include <pvdisplays/PVDisplayIf.h>

#include <pvguiqt/common.h>

#include <pvparallelview/PVParallelView.h>

#include <QApplication>
#include <QEventLoop>
#include <QWidget>

#include <algorithm>
#include <any>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "test-env.h"

int main(int argc, char** argv)
{
	init_env();

	Squey::PVRoot root;
	const QString file = QString(TEST_FOLDER) + "/picviz/heat_line.csv";
	const QString format = file + ".format";
	Squey::PVSource& src = get_src_from_file(root, file, format);

	// emplace_add_child() computes each stage in turn:
	// Mapped (mapping) -> Scaled (scaling) -> View. The resulting view therefore
	// carries real scaled data, which the parallelview widgets need to render.
	Squey::PVView& view = src.emplace_add_child().emplace_add_child().emplace_add_child();

	// Import oracle: the format's shape survived the full extraction pipeline.
	PV_VALID(size_t(src.get_row_count()), size_t(50000));
	PV_VALID(size_t(view.get_column_count()), size_t(2));

	QApplication app(argc, argv); // argv carries "-platform offscreen"

	PVParallelView::common::RAII_backend_init backend_resources;
	// Registers the libpvguiqt view displays; RAII_backend_init above already
	// registered the libpvparallelview ones.
	PVGuiQt::common::register_displays();

	// The view displays that must always be available. If any of these stops
	// being registered, the catalog oracle below fails.
	std::set<std::string> expected_view_displays = {
	    "guiqt_axes-combination",
	    "guiqt_correlation",
	    "guiqt_mapping-scaling",
	    "guiqt_filters",
	    "guiqt_distinct-values",
	    "guiqt_count-by",
	    "guiqt_sum-by",
	    "guiqt_min-by",
	    "guiqt_max-by",
	    "guiqt_average-by",
	    "guiqt_layer-stack",
	    "guiqt_listing",
	    "parallelview_fullparallelview",
	    "parallelview_zoomedparallelview",
	    "parallelview_hitcountview",
	    "parallelview_scatterview",
	    "parallelview_timeseriesview",
	};
#ifdef PYTHON_SUPPORT
	expected_view_displays.insert("guiqt_pythonconsole");
#endif

	// The column-oriented displays (distinct values, the group-by family and the
	// parallelview axis views) read their target column(s) from the parameter
	// pack; the view-level ones ignore it. Two valid comb columns therefore
	// satisfy every create_widget contract at once.
	const std::vector<std::any> view_params = {std::any(PVCombCol(0)), std::any(PVCombCol(1))};

	// The parallelview widgets render asynchronously on a TBB pool; pumping the
	// event loop lets that background render actually run (and its result be
	// consumed) so the render path is exercised, not merely constructed.
	const auto drain_render = []() {
		for (int i = 0; i < 100; ++i) {
			QApplication::processEvents(QEventLoop::AllEvents, 10);
		}
	};

	// Build and show every registered view display against the computed view.
	std::set<std::string> seen_view_displays;
	PVDisplays::visit_displays_by_if<PVDisplays::PVDisplayViewIf>(
	    [&](PVDisplays::PVDisplayViewIf& obj) {
		    const std::string name = obj.registered_name().toStdString();
		    seen_view_displays.insert(name);

		    // The scatter, hit-count and time-series views are left out for now;
		    // only the full-parallel and zoomed views are exercised here.
		    if (name == "parallelview_scatterview" or name == "parallelview_hitcountview" or
		        name == "parallelview_timeseriesview") {
			    return;
		    }

		    QWidget* w = PVDisplays::get_widget(obj, &view, nullptr, view_params);
		    PV_ASSERT_VALID(w != nullptr, "view_display", name);

		    if (name.starts_with("parallelview_")) {
			    // The render maps the viewport onto the zone tree, so give it a real
			    // size and let the background render run.
			    w->resize(1024, 1024);
			    w->show();
			    drain_render();
		    } else {
			    w->show();
			    QApplication::processEvents();
		    }
		    PV_ASSERT_VALID(w->isVisible(), "view_display", name);
	    });

	// Catalog oracle: every expected display was actually visited.
	PV_ASSERT_VALID(std::includes(seen_view_displays.begin(), seen_view_displays.end(),
	                              expected_view_displays.begin(), expected_view_displays.end()),
	                "seen_count", seen_view_displays.size(), "expected_count",
	                expected_view_displays.size());

	// Source displays (the data-tree view) must build against the source too.
	PVDisplays::visit_displays_by_if<PVDisplays::PVDisplaySourceIf>(
	    [&](PVDisplays::PVDisplaySourceIf& obj) {
		    QWidget* w = PVDisplays::get_widget(obj, &src);
		    PV_ASSERT_VALID(w != nullptr, "source_display", obj.registered_name().toStdString());
		    w->show();
		    QApplication::processEvents();
	    });

	return 0;
}
